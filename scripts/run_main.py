import networkx as nx
import numpy as np
import math
import os

from peo_pycuda.chordal_gen import generateChordalGraph

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule

cuda_code = """
#include <stdio.h>
#include <scan.cu>
#include <stratify_none.cu>
#include <stratify_high_degree.cu>
#include <stratify_low_degree.cu>

extern "C" {

__global__ void stratify(double *numbering, float *roots, int *indptr, int *indices, double delta, int n);

__host__ __device__ void print_array(float *a, int n)
{
    for(int i=0; i<n; i++)
        printf("%f ", a[i]);
    printf("\\n");
}

__host__ __device__ void parallel_prefix(float *d_idata, float *d_odata, int num_elements)
{

    num_elements += 1;
    float** g_scanBlockSums;

    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = num_elements;

    int level = 0;

    do
    {
        unsigned int numBlocks =
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    g_scanBlockSums = (float**) malloc(level * sizeof(float*));

    numElts = num_elements;
    level = 0;

    do
    {
        unsigned int numBlocks =
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            cudaMalloc((void**) &g_scanBlockSums[level++],
                                      numBlocks * sizeof(float));
        }
        numElts = numBlocks;
    } while (numElts > 1);

    prescanArrayRecursive(d_odata, d_idata, num_elements, 0, g_scanBlockSums);

}

__global__ void init_array(float *arr, float val)
{
    const int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    arr[i] = val;

}

__global__ void split_classes(double *numbering, int *indptr, int *indices, float *mask, float *roots, float *changes)
{
    const int i = threadIdx.x;
    if(mask[i] == 0){
        roots[i] = -1;
        return;
    }
    int min = roots[i];

    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(mask[indices[j]] == 1 && numbering[i] == numbering[indices[j]] && roots[indices[j]] < min){
            min = roots[indices[j]];
        }
    }

    if(min != roots[i]){
        roots[i] = min;
        changes[i] += 1;
    }
}

__host__ __device__ void get_class_components(double *numbering, int *indptr, int *indices, float *mask, int n, float *roots)
{
    float *changes, *sum;
    
    cudaMalloc((void**)&changes, sizeof(float) * n);
    cudaMalloc((void**)&sum, sizeof(float) * n);
    
    do{
        init_array<<< 1, n >>>(changes, 0);
        cudaDeviceSynchronize();
        split_classes<<< 1, n >>>(numbering, indptr, indices, mask, roots, changes);
        cudaDeviceSynchronize();
        parallel_prefix(changes, sum, n);
        cudaDeviceSynchronize();
    }while(sum[n] > 0);
    
}

__global__ void get_class_components_global(double *numbering, int *indptr, int *indices, float *mask, int n, float *roots)
{

    get_class_components(numbering, indptr, indices, mask, n, roots);
    
}

__global__ void spanning_tree_depth(int *indptr, int *indices, float *level, float *in_component, int *neighbors, int curr_level)
{
    const int i = threadIdx.x;
    int curr_node = neighbors[i];
    if(level[curr_node] > 0 || in_component[curr_node] == 0)
        return;
    level[curr_node] = curr_level;

    int j = indptr[curr_node];
    int num_neighbors = indptr[curr_node+1] - indptr[curr_node];
    if(num_neighbors > 0){
        __syncthreads();
        spanning_tree_depth<<< 1, num_neighbors >>>(indptr, indices, level, in_component, indices+j*sizeof(int), curr_level+1);
        cudaDeviceSynchronize();
    }
}


//outputs level of depth forming a spanning tree for a given root in component. the level-node index pair gives a unique
//depth ordering for each node in the component

__host__ __device__ void spanning_tree_numbering(int *indptr, int *indices, float *in_component, float *level, int root, int n)
{
    init_array<<< 1, n >>>(level, 0);
    cudaDeviceSynchronize();
    level[root] = 1;
    int j = indptr[root];
    int num_neighbors = indptr[root + 1] - indptr[root];
    spanning_tree_depth<<< 1, num_neighbors >>>(indptr, indices, level, in_component, indices+j*sizeof(int), 2);
    cudaDeviceSynchronize();
    
}

__global__ void compute_component_sizes(float *roots, float *sizes)
{
    const int i = threadIdx.x;
    int root;
    root = roots[i];
    sizes[root] += 1;
}


__global__ void richer_neighbors(double *numbering, float *roots, int *indptr, int *indices, int root, float c, float *is_richer_neighbor, float *high_degree, float *neighbors_in_c)
{
    const int i = threadIdx.x;
    is_richer_neighbor[i] = 0;
    high_degree[i] = 0;
    neighbors_in_c[i] = 0;
    if(roots[i] == root) return;

    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(numbering[i] > numbering[indices[j]] && roots[indices[j]] == root){
            is_richer_neighbor[i] = 1;
            neighbors_in_c[i] += 1;
        }
    }

    if(neighbors_in_c[i] >= 2 / 5 * c){
        high_degree[i] = 1;
    }
}

__global__ void in_class(double *numbering, float *roots, int *indptr, int *indices, int c, float *is_class_component)
{
    const int i = threadIdx.x;
    is_class_component[i] = 0;
    if(roots[i] == c) is_class_component[i] = 1;
}

__global__ void in_class_special(double *numbering, float *roots, int *indptr, int *indices, int c, float *is_class_component)
{
    const int i = threadIdx.x;
    is_class_component[i] = -1;
    if(roots[i] == c) is_class_component[i] = c;
}

__global__ void is_clique(float *in_component, int *indptr, int *indices, int n, float c, float *full_connected)
{
    const int i = threadIdx.x;
    full_connected[i] = 0;
    if(in_component[i] == 0) return;
    
    int d = 0;
    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(in_component[indices[j]] == 1){
            d += 1;
        }
    }
    
    if(d >= c-1){
        full_connected[i] = 1;
    }
}

__global__ void sum_array_to_list(float *sums, float *list)
{
    const int i = threadIdx.x;
    
    if(sums[i+1] == sums[i]) return;
    
    list[(int)sums[i+1] - 1] = i;
}

__global__ void add_i(double *numbering, float *D_sum, int *indptr, int *indices, int n)
{
    const int i = threadIdx.x;
    
    if(D_sum[i+1] == D_sum[i]) return;
    
    numbering[i] += D_sum[i+1];
}

__global__ void difference(float *a, float *b, float *r)
{
    const int i = threadIdx.x;
    r[i] = a[i] * (1 - b[i]);
}


__global__ void find_first(float *a, int *first)
{
    const int i = threadIdx.x;
    if(a[i+1] == 1 && a[i] == 0) *first = i;
}

__global__ void inc_delta(double *numbering, float *other_array, double delta)
{
    const int i = threadIdx.x;
    if(other_array[i] == 1) numbering[i] += delta;
}

__global__ void find_common_neighbors(float *is_class_component, int *indptr, int *indices, int f, int s, float *r)
{
    const int i = threadIdx.x;
    r[i] = 0;
    if(is_class_component[i] == 0) return;
    
    int d = 0;
    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(indices[j] == f || indices[j] == s){
            d += 1;
        }
    }
    
    if(d == 2) r[i] = 1;
}

__device__ void stratify_none(double *numbering, float *is_class_component, int *indptr, int *indices, double delta, int n, float c)
{
    float *D, *C_D, *D_clique, *D_diff, *D_diff_first_neigh, *D_diff_first_neigh_diff, *common_neighbors, *C_D_components;
    float *D_sum, *D_clique_sum, *D_diff_sum, *D_diff_first_neigh_sum, *common_neighbors_sum, *C_D_components_sizes;
    float rolling_sum;
    int i, j, flag, c_root;

    int *first, *second;
    cudaMalloc((void**)&first, sizeof(int));
    cudaMalloc((void**)&second, sizeof(int));

    unsigned int pps_arr_size  = (n+1)*sizeof(float);
    cudaMalloc((void**)&D, pps_arr_size);
    cudaMalloc((void**)&C_D, pps_arr_size);
    cudaMalloc((void**)&C_D_components, pps_arr_size);
    cudaMalloc((void**)&D_clique, pps_arr_size);
    cudaMalloc((void**)&D_sum, pps_arr_size);
    cudaMalloc((void**)&D_clique_sum, pps_arr_size);
    cudaMalloc((void**)&D_diff, pps_arr_size);
    cudaMalloc((void**)&D_diff_sum, pps_arr_size);
    cudaMalloc((void**)&D_diff_first_neigh, pps_arr_size);
    cudaMalloc((void**)&D_diff_first_neigh_sum, pps_arr_size);
    cudaMalloc((void**)&D_diff_first_neigh_diff, pps_arr_size);
    cudaMalloc((void**)&common_neighbors, pps_arr_size);
    cudaMalloc((void**)&common_neighbors_sum, pps_arr_size);


    cudaMalloc((void**)&C_D_components_sizes, pps_arr_size);
    init_array<<< 1, n >>>(C_D_components_sizes, 0);


    stratify_none_getD<<< 1, n >>>(is_class_component, indptr, indices, n, c, D);
    cudaDeviceSynchronize();
    difference<<< 1, n >>>(is_class_component, D, C_D);
    cudaDeviceSynchronize();

    get_class_components(numbering, indptr, indices, C_D, n, C_D_components);
    compute_component_sizes<<< 1, n >>>(C_D_components, C_D_components_sizes);
    cudaDeviceSynchronize();
    for(i = 0, rolling_sum = 0, flag = 0; i < n && flag == 0; i++)
        if(C_D_components_sizes[i] >0){
            if(C_D_components_sizes[i] > c* 4/5){
                flag = 100;
                c_root = i;
            }else{
                rolling_sum += C_D_components_sizes[i];
                if(rolling_sum > c *1/5)
                    flag = 1;
                // raise flag, no component can be > 4/5
            }
        }

    if(flag>1){ //component exists
        float *level, *adjacencies, *in_component;
        cudaMalloc((void**)&adjacencies, n*n*sizeof(float));
        cudaMalloc((void**)&level, pps_arr_size);
        cudaMalloc((void**)&in_component, pps_arr_size);
        in_class<<< 1, n >>>(numbering, C_D_components, indptr, indices, c_root, in_component);
        init_array<<< n, n >>>(adjacencies, 0);
        cudaDeviceSynchronize();
        compute_adjacent_nodes<<< 1, n >>>(indptr, indices, is_class_component, in_component, adjacencies, n);
        //add_self<<< 1, n >>>(is_class_component, in_component, adjacencies, n);
        spanning_tree_numbering(indptr, indices, in_component, level, c_root, n);

        float *arr_even, *arr_odd, *curr_array, *other_array, *tmp_arr_pointer, *sum;
        cudaMalloc((void**)&arr_odd, n*sizeof(float));
        cudaMalloc((void**)&arr_even, n*sizeof(float));
        cudaMalloc((void**)&sum, n*sizeof(float));
        int current_depth;
        flag = 0;
        cudaDeviceSynchronize();
        other_array = adjacencies + c_root*sizeof(float);
        curr_array = arr_even;
        tmp_arr_pointer = arr_odd;
        other_array[c_root] = 1;

        //Flip between even and odd arrays instead of saving old values. When we go above the threshold, we use the other
        //array for indices
        for(i = 0, j = 0, current_depth = 2; flag == 0; i++){
            if(level[i]==current_depth){
                if(j > 0){
                    other_array = curr_array;
                    curr_array = tmp_arr_pointer;
                    tmp_arr_pointer = other_array; // for next cycle - needed due to first declaration used to skip copying
                }
                logic_or<<< 1, n >>>(other_array, adjacencies+i*sizeof(float), curr_array);
                cudaDeviceSynchronize();
                parallel_prefix(curr_array, sum, n);
                cudaDeviceSynchronize();
                if(sum[n] > c * 4/5)
                    flag = 1;
                j++;
            }
            if(flag == 0 && i == n-1){
                i = 0; // 0 is either c_root or not in component, increment irrelevant
                current_depth +=1;
            }
        }

        inc_delta<<< 1, n >>>(numbering, other_array, delta);
        return;
    }
    //number of members of D
    parallel_prefix(D, D_sum, n);
    //get number of neighbors in D of nodes in D
    is_clique<<< 1, n >>>(D, indptr, indices, n, c, D_clique);
    cudaDeviceSynchronize();
    //check if they are all connected
    parallel_prefix(D_clique, D_clique_sum, n);
    cudaDeviceSynchronize();
    if(D_clique_sum[n] == D_sum[n]){
        add_i<<< 1, n >>>(numbering, D_sum, indptr, indices, n);
    }else{
        difference<<< 1, n >>>(D, D_clique, D_diff);
        cudaDeviceSynchronize();
        parallel_prefix(D_diff, D_diff_sum, n);
        cudaDeviceSynchronize();
        find_first<<< 1, n >>>(D_diff_sum, first);
        cudaDeviceSynchronize();
        D_diff_first_neigh[*first] = 1;
        for(int j = indptr[*first]; j < indptr[*first + 1]; j++){
            if(D_diff[indices[j]] == 1){
                D_diff_first_neigh[indices[j]] = 1;
            }
        }
        difference<<< 1, n >>>(D_diff, D_diff_first_neigh, D_diff_first_neigh_diff);
        cudaDeviceSynchronize();
        parallel_prefix(D_diff_first_neigh_diff, D_diff_first_neigh_sum, n);
        cudaDeviceSynchronize();
        find_first<<< 1, n >>>(D_diff_first_neigh_sum, second);
        cudaDeviceSynchronize();
        find_common_neighbors<<< 1, n >>>(D, indptr, indices, *first, *second, common_neighbors);
        cudaDeviceSynchronize();
        parallel_prefix(common_neighbors, common_neighbors_sum, n);
        cudaDeviceSynchronize();
        add_i<<< 1, n >>>(numbering, common_neighbors_sum, indptr, indices, n);
    }
}



__device__ void stratify_high_degree(double *numbering, float *is_class_component, int *indptr, int *indices, double delta, int n, float *is_richer_neighbor, float irn_num, float icc_num)
{
    float *adjacencies;
    cudaMalloc((void**)&adjacencies, n*n*sizeof(float));
    init_array<<< n, n >>>(adjacencies, 0);
    cudaDeviceSynchronize();
    compute_adjacent_nodes<<< 1, n >>>(indptr, indices, is_class_component, is_richer_neighbor, adjacencies, n);
    cudaDeviceSynchronize();

    float *arr_even, *arr_odd, *curr_array, *other_array, *sum;
    cudaMalloc((void**)&arr_odd, n*sizeof(float));
    init_array<<< 1, n >>>(arr_odd, 1); //this array will be the first to be used for the logical and, we will write into arr_even
    cudaMalloc((void**)&arr_even, n*sizeof(float));
    cudaMalloc((void**)&sum, n*sizeof(float));
    int i, j, flag;
    flag = 0;
    cudaDeviceSynchronize();

    //Flip between even and odd arrays instead of saving old values. When we go below the threshold, we use the other
    //array for indices
    for(i = 0, j = 0; i < n && flag == 0; i++){
        if(is_richer_neighbor[i]){
            if(j%2 == 0){
                curr_array = arr_even;
                other_array = arr_odd;
            }else{
                curr_array = arr_odd;
                other_array = arr_even;
            }
            logic_and<<< 1, n >>>(other_array, adjacencies+i*sizeof(float), curr_array);
            cudaDeviceSynchronize();
            if(j > 0){
                parallel_prefix(curr_array, sum, n);
                cudaDeviceSynchronize();
                if(sum[n] < icc_num/5)
                    flag = 1;
            }
            j++;
        }
    }

    inc_delta<<< 1, n >>>(numbering, other_array, delta);
    
    if(j == irn_num){
        float *C1_components, *C1_components_sizes, *C1, *component_size;
        cudaMalloc((void**)&component_size, n*sizeof(float));
        cudaMalloc((void**)&C1, n*sizeof(float));
        cudaMalloc((void**)&C1_components, n*sizeof(float));
        cudaMalloc((void**)&C1_components_sizes, n*sizeof(float));
        init_array<<< 1, n >>>(component_size, 0);
        get_class_components(numbering, indptr, indices, other_array, n, C1_components);
        compute_component_sizes<<< 1, n >>>(C1_components, C1_components_sizes);
        cudaDeviceSynchronize();
        int max_size_root = 0;
        for(i = 0; i < n; i++){
            if(C1_components_sizes[i] >0){
                if(C1_components_sizes[i] > C1_components_sizes[max_size_root]){
                    max_size_root = i;
                }
            }
        }
        in_class<<< 1, n >>>(numbering, C1_components, indptr, indices, max_size_root, C1);
        cudaDeviceSynchronize();
        stratify_none(numbering, C1, indptr, indices, delta / 2, n, C1_components_sizes[max_size_root]);
    }


}

__device__ void stratify_low_degree(double *numbering, float *is_class_component, int *indptr, int *indices, double delta, int n, float *is_richer_neighbor, float c)
{
    float *D, *CuB, *CuB_D, *CuB_D_components, *CuB_D_components_sum;
    int *b_root;
    int i, j, flag;

    unsigned int pps_arr_size  = (n+1)*sizeof(float);
    cudaMalloc((void**)&b_root, sizeof(int));
    cudaMalloc((void**)&D, pps_arr_size);
    cudaMalloc((void**)&CuB, pps_arr_size);
    cudaMalloc((void**)&CuB_D, pps_arr_size);
    cudaMalloc((void**)&CuB_D_components, pps_arr_size);
    cudaMalloc((void**)&CuB_D_components_sum, pps_arr_size);

    float *level, *adjacencies, *in_component;
    cudaMalloc((void**)&adjacencies, n*n*sizeof(float));
    cudaMalloc((void**)&level, pps_arr_size);
    cudaMalloc((void**)&in_component, pps_arr_size);

    logic_or<<< 1, n >>>(is_richer_neighbor, is_class_component, CuB);
    cudaDeviceSynchronize();
    
    stratify_lowdegree_getD<<< 1, n >>>(CuB, is_class_component, indptr, indices, n, c, D);
    cudaDeviceSynchronize();
    
    difference<<< 1, n >>>(CuB, D, CuB_D);
    cudaDeviceSynchronize();

    get_class_components(numbering, indptr, indices, CuB_D, n, CuB_D_components);
    cudaDeviceSynchronize();

    parallel_prefix(CuB_D_components, CuB_D_components_sum, n);
    cudaDeviceSynchronize();
    find_first<<< 1, n >>>(CuB_D_components_sum, b_root);
    cudaDeviceSynchronize();

    printf("CuB-D components\\n");
    print_array(CuB_D_components, n);
    in_class<<< 1, n >>>(numbering, CuB_D_components, indptr, indices, *b_root, in_component);
    init_array<<< n, n >>>(adjacencies, 0);
    cudaDeviceSynchronize();
    compute_adjacent_nodes<<< 1, n >>>(indptr, indices, is_class_component, in_component, adjacencies, n);
    //add_self<<< 1, n >>>(is_class_component, in_component, adjacencies, n);
    
    spanning_tree_numbering(indptr, indices, in_component, level, *b_root, n);

    float *arr_even, *arr_odd, *curr_array, *other_array, *tmp_arr_pointer, *sum;
    cudaMalloc((void**)&arr_odd, n*sizeof(float));
    cudaMalloc((void**)&arr_even, n*sizeof(float));
    cudaMalloc((void**)&sum, n*sizeof(float));
    int current_depth;
    flag = 0;
    cudaDeviceSynchronize();
    other_array = adjacencies + (*b_root) * sizeof(float);
    curr_array = arr_even;
    tmp_arr_pointer = arr_odd;
    other_array[*b_root] = 1;

    //Flip between even and odd arrays instead of saving old values. When we go above the threshold, we use the other
    //array for indices
    for(i = 0, j = 0, current_depth = 2; flag == 0; i++){
        if(level[i]==current_depth){
            if(j > 0){
                other_array = curr_array;
                curr_array = tmp_arr_pointer;
                tmp_arr_pointer = other_array; // for next cycle - needed due to first declaration used to skip copying
            }
            logic_or<<< 1, n >>>(other_array, adjacencies+i*sizeof(float), curr_array);
            cudaDeviceSynchronize();
            parallel_prefix(curr_array, sum, n);
            cudaDeviceSynchronize();
            if(sum[n] > c * 4/5)
                flag = 1;
            j++;
        }
        if(flag == 0 && i == n-1){
            i = -1;
            current_depth +=1;
        }
    }

    logic_and<<< 1, n >>>(other_array, is_class_component, other_array);
    inc_delta<<< 1, n >>>(numbering, other_array, delta);

    float *C1_components, *C1_components_sizes, *C1, *component_size, *C_A;
    cudaMalloc((void**)&component_size, n*sizeof(float));
    cudaMalloc((void**)&C1, n*sizeof(float));
    cudaMalloc((void**)&C1_components, n*sizeof(float));
    cudaMalloc((void**)&C_A, n*sizeof(float));
    cudaMalloc((void**)&C1_components_sizes, n*sizeof(float));
    init_array<<< 1, n >>>(component_size, 0);
    difference<<< 1, n >>>(is_class_component, other_array, C_A);
    cudaDeviceSynchronize();
    get_class_components(numbering, indptr, indices, C_A, n, C1_components);
    cudaDeviceSynchronize();
    compute_component_sizes<<< 1, n >>>(C1_components, C1_components_sizes);
    cudaDeviceSynchronize();
    int max_size_root = 0;
    for(i = 0; i < n; i++){
        if(C1_components_sizes[i] >0){
            if(C1_components_sizes[i] > C1_components_sizes[max_size_root]){
                max_size_root = i;
            }
        }
    }

    if(C1_components_sizes[max_size_root] > c * 4/5){
        in_class_special<<< 1, n >>>(numbering, C1_components, indptr, indices, max_size_root, C1);
        cudaDeviceSynchronize();
        stratify<<< 1, n >>>(numbering, C1, indptr, indices, delta / 2, n);
        cudaDeviceSynchronize();
    }

}

__global__ void stratify(double *numbering, float *roots, int *indptr, int *indices, double delta, int n)
{
    const int i = threadIdx.x;
    if(roots[i] != i) return;

    float *is_richer_neighbor, *high_degree, *is_class_component, *neighbors_in_c;
    float *irn_sum, *hd_sum, *icc_sum, *nic_sum;

    unsigned int pps_arr_size  = (n+1)*sizeof(float);
    cudaMalloc((void**)&is_richer_neighbor, pps_arr_size);
    cudaMalloc((void**)&high_degree, pps_arr_size);
    cudaMalloc((void**)&is_class_component, pps_arr_size);
    cudaMalloc((void**)&neighbors_in_c, pps_arr_size);
    cudaMalloc((void**)&irn_sum, pps_arr_size);
    cudaMalloc((void**)&hd_sum, pps_arr_size);
    cudaMalloc((void**)&icc_sum, pps_arr_size);
    cudaMalloc((void**)&nic_sum, pps_arr_size);


    in_class<<< 1, n >>>(numbering, roots, indptr, indices, roots[i], is_class_component);
    cudaDeviceSynchronize();
    parallel_prefix(is_class_component, icc_sum, n);
    cudaDeviceSynchronize();
    //if(icc_sum[n] <= 1) return;
    
    richer_neighbors<<< 1, n >>>(numbering, roots, indptr, indices, roots[i], icc_sum[n], is_richer_neighbor, high_degree, neighbors_in_c);
    cudaDeviceSynchronize();
    parallel_prefix(is_richer_neighbor, irn_sum, n);
    cudaDeviceSynchronize();
    /*
    print_array(is_richer_neighbor, n);
    print_array(high_degree, n);
    print_array(neighbors_in_c, n);
    */
    if(irn_sum[n] == 0)
        stratify_none(numbering, is_class_component, indptr, indices, delta, n, icc_sum[n]);
    else{
        parallel_prefix(high_degree, hd_sum, n);
        cudaDeviceSynchronize();
        if(hd_sum[n] >= irn_sum[n])
            stratify_high_degree(numbering, is_class_component, indptr, indices, delta, n, is_richer_neighbor, irn_sum[n], icc_sum[n]);
        else
            stratify_low_degree(numbering, is_class_component, indptr, indices, delta, n, is_richer_neighbor, icc_sum[n]);
    }
    cudaDeviceSynchronize();

}

}

"""

cuda_module = DynamicSourceModule(cuda_code, include_dirs=[os.path.join(os.getcwd(), '..', 'lib')], no_extern_c=True)
stratify = cuda_module.get_function("stratify")
split_classes = cuda_module.get_function("get_class_components_global")

N = 50
DENSITY = 0.5

G = generateChordalGraph(N, DENSITY, debug=False)

Gcsr = nx.to_scipy_sparse_matrix(G)
numbering = np.zeros(N, dtype=np.float64)

delta = 8 ** math.ceil(math.log(N, 5/4))

extra_space = int(N / 16 + N / 16**2 + 1)
print(Gcsr)
unique_numberings = np.unique(numbering)
while len(unique_numberings) < len(numbering) and delta >= 1:
    roots = np.arange(N, dtype=np.float32)
    split_classes(cuda.In(numbering), cuda.In(Gcsr.indptr), cuda.In(Gcsr.indices), cuda.In(np.ones(N, dtype=np.float32)), np.int32(N), cuda.InOut(roots), block=(1, 1, 1), shared=8*(N+extra_space+10))
    stratify(cuda.InOut(numbering), cuda.In(roots), cuda.In(Gcsr.indptr), cuda.In(Gcsr.indices), np.float64(delta), np.int32(N), block=(N, 1, 1), shared=8*(N+extra_space+10))
    delta /= 8
    unique_numberings = np.unique(numbering)
print(numbering)

if(len(unique_numberings) == len(numbering)):
    print("CHORDAL")
else:
    print("NOT CHORDAL")