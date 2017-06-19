import networkx as nx
import numpy as np
import math
import os
import time
import matplotlib.pyplot as plt

from peo_pycuda.chordal_gen import generateChordalGraph, generateGraph

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.compiler import DynamicSourceModule

def delta_divide(delta1, delta2):
    dec = 3
    if delta2 == 0:
        delta2 = np.uint64(2 ** 63)
    while dec > 0 and delta1 > 0:
        delta1 = np.uint64(delta2 / 2)
        dec -= 1
    while dec > 0 and delta2 > 0:
        delta2 = np.uint64(delta2 / 2)
        dec -= 1
    if delta1 > 0:
        delta2 = np.uint64(0)
    return delta1, delta2

cuda_code = """
#include <stdio.h>
#include <scan.cu>
#include <mainlib.cu>
#include <stratify_none.cu>
#include <stratify_high_degree.cu>
#include <stratify_low_degree.cu>

extern "C" {

__global__ void stratify(my_uint128 *numbering, float *roots, int *indptr, int *indices, unsigned long long int delta1, unsigned long long int delta2, int n);


// needed for pycuda call, since get_class_components is used both inside and outside stratify calls
__global__ void get_class_components_global(my_uint128 *numbering, int *indptr, int *indices, float *mask, int n, float *roots)
{

    get_class_components(numbering, indptr, indices, mask, n, roots);
    
}


// stratification for components with no richer neighbors
__device__ void stratify_none(my_uint128 *numbering, float *is_class_component, int *indptr, int *indices, my_uint128 delta, int n, float c)
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

    // Here we are searching for a component with size > 4/5 C
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

    if(flag > 1){ //component exists
        float *level, *adjacencies, *in_component;
        cudaMalloc((void**)&adjacencies, n * n * sizeof(float));
        cudaMalloc((void**)&level, pps_arr_size);
        cudaMalloc((void**)&in_component, pps_arr_size);
        in_class<<< 1, n >>>(C_D_components, c_root, in_component);
        init_array<<< n, n >>>(adjacencies, 0);
        cudaDeviceSynchronize();
        compute_adjacent_nodes<<< 1, n >>>(indptr, indices, is_class_component, in_component, adjacencies, n);
        spanning_tree_numbering(indptr, indices, in_component, level, c_root, n);

        float *arr_even, *arr_odd, *curr_array, *other_array, *tmp_arr_pointer, *sum;
        cudaMalloc((void**)&arr_odd, pps_arr_size);
        cudaMalloc((void**)&arr_even, pps_arr_size);
        cudaMalloc((void**)&sum, pps_arr_size);
        int current_depth;
        flag = 0;
        cudaDeviceSynchronize();
        other_array = adjacencies + c_root*sizeof(float);
        curr_array = arr_even;
        tmp_arr_pointer = arr_odd;
        other_array[c_root] = 1;

        //Searching for the maximal set of indices for stratification, by using logical ors for successive cycles
        //Flip between even and odd arrays instead of saving old values. When we go above the threshold, we use the other
        //array for indices, since it corresponds to the maximal set before the condition was violated
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
        cudaFree(adjacencies);
        cudaFree(level);
        cudaFree(in_component);
        cudaFree(arr_odd);
        cudaFree(arr_even);
        cudaFree(sum);
    }else{
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
            // We generate the set of not completely connected nodes in D, select the first node in this set, find
            // its non-neighbors, select the first non-neighbor, and then find the common neighbors of this set
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
    cudaFree(D);
    cudaFree(C_D);
    cudaFree(C_D_components);
    cudaFree(D_clique);
    cudaFree(D_sum);
    cudaFree(D_clique_sum);
    cudaFree(D_diff);
    cudaFree(D_diff_sum);
    cudaFree(D_diff_first_neigh);
    cudaFree(D_diff_first_neigh_sum);
    cudaFree(D_diff_first_neigh_diff);
    cudaFree(common_neighbors);
    cudaFree(common_neighbors_sum);
    cudaFree(C_D_components_sizes);
    cudaFree(first);
    cudaFree(second);
}

// Stratification for components where every richer node has at least 2/5*|C| neighbors in C
__device__ void stratify_high_degree(my_uint128 *numbering, float *is_class_component, int *indptr, int *indices, my_uint128 delta, int n, float *is_richer_neighbor, float irn_num, float icc_num)
{
    unsigned int pps_arr_size  = (n+1)*sizeof(float);
    float *adjacencies;
    cudaMalloc((void**)&adjacencies, n * n * sizeof(float));
    init_array<<< n, n >>>(adjacencies, 0);
    cudaDeviceSynchronize();
    compute_adjacent_nodes<<< 1, n >>>(indptr, indices, is_class_component, is_richer_neighbor, adjacencies, n);
    cudaDeviceSynchronize();

    float *arr_even, *arr_odd, *curr_array, *other_array, *sum;
    cudaMalloc((void**)&arr_odd, pps_arr_size);
    init_array<<< 1, n >>>(arr_odd, 1); //this array will be the first to be used for the logical and, we will write into arr_even
    cudaMalloc((void**)&arr_even, pps_arr_size);
    cudaMalloc((void**)&sum, pps_arr_size);
    int i, j, flag;
    flag = 0;
    cudaDeviceSynchronize();

    //Searching for the maximal set of indices for stratification, by using logical ands for successive cycles
    //Flip between even and odd arrays instead of saving old values. When we go below the threshold, we use the other
    //array for indices
    for(i = 0, j = 0; i < n && flag == 0; i++){
        if(is_richer_neighbor[i]){
            if(j % 2 == 0){
                curr_array = arr_even;
                other_array = arr_odd;
            }else{
                curr_array = arr_odd;
                other_array = arr_even;
            }
            logic_and<<< 1, n >>>(other_array, adjacencies + i * sizeof(float), curr_array);
            cudaDeviceSynchronize();
            if(j > 0){
                parallel_prefix(curr_array, sum, n);
                cudaDeviceSynchronize();
                if(sum[n] < icc_num / 5)
                    flag = 1;
            }
            j++;
        }
    }

    inc_delta<<< 1, n >>>(numbering, other_array, delta);
    
    if(j == irn_num){
        // we search for the maximal subcomponent of set C1, and then call stratify_none on it
        float *C1_components, *C1_components_sizes, *C1, *component_size;
        cudaMalloc((void**)&component_size, pps_arr_size);
        cudaMalloc((void**)&C1, pps_arr_size);
        cudaMalloc((void**)&C1_components, pps_arr_size);
        cudaMalloc((void**)&C1_components_sizes, pps_arr_size);
        init_array<<< 1, n >>>(component_size, 0);
        get_class_components(numbering, indptr, indices, other_array, n, C1_components);
        compute_component_sizes<<< 1, n >>>(C1_components, C1_components_sizes);
        cudaDeviceSynchronize();
        int max_size_root = 0;
        for(i = 0; i < n; i++){
            if(C1_components_sizes[i] > 0){
                if(C1_components_sizes[i] > C1_components_sizes[max_size_root]){
                    max_size_root = i;
                }
            }
        }
        in_class<<< 1, n >>>(C1_components, max_size_root, C1);
        cudaDeviceSynchronize();
        stratify_none(numbering, C1, indptr, indices, shl_my_uint128(delta, -1), n, C1_components_sizes[max_size_root]);
        cudaFree(component_size);
        cudaFree(C1);
        cudaFree(C1_components);
        cudaFree(C1_components_sizes);
    }

    cudaFree(arr_even);
    cudaFree(sum);
    cudaFree(arr_odd);
    cudaFree(adjacencies);

}

// Stratification for components where exists a richer node that has less than 2/5*|C| neighbors in C
__device__ void stratify_low_degree(my_uint128 *numbering, float *is_class_component, int *indptr, int *indices, my_uint128 delta, int n, float *is_richer_neighbor, float c)
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

    in_class<<< 1, n >>>(CuB_D_components, *b_root, in_component);
    init_array<<< n, n >>>(adjacencies, 0);
    cudaDeviceSynchronize();
    compute_adjacent_nodes<<< 1, n >>>(indptr, indices, is_class_component, in_component, adjacencies, n);
    
    spanning_tree_numbering(indptr, indices, in_component, level, *b_root, n);

    float *arr_even, *arr_odd, *curr_array, *other_array, *tmp_arr_pointer, *sum;
    cudaMalloc((void**)&arr_odd, pps_arr_size);
    cudaMalloc((void**)&arr_even, pps_arr_size);
    cudaMalloc((void**)&sum, pps_arr_size);
    int current_depth;
    flag = 0;
    cudaDeviceSynchronize();
    other_array = adjacencies + (*b_root) * sizeof(float);
    curr_array = arr_even;
    tmp_arr_pointer = arr_odd;
    other_array[*b_root] = 1;

    //Searching for the maximal set of indices for stratification, by using logical ors for successive cycles
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
    cudaMalloc((void**)&component_size, pps_arr_size);
    cudaMalloc((void**)&C1, pps_arr_size);
    cudaMalloc((void**)&C1_components, pps_arr_size);
    cudaMalloc((void**)&C_A, pps_arr_size);
    cudaMalloc((void**)&C1_components_sizes, pps_arr_size);
    init_array<<< 1, n >>>(component_size, 0);
    difference<<< 1, n >>>(is_class_component, other_array, C_A);
    cudaDeviceSynchronize();
    get_class_components(numbering, indptr, indices, C_A, n, C1_components);
    cudaDeviceSynchronize();
    compute_component_sizes<<< 1, n >>>(C1_components, C1_components_sizes);
    cudaDeviceSynchronize();
    
    // We have to find the largest component C1 of C - Aj (we find the most frequent root)
    int max_size_root = 0;
    for(i = 0; i < n; i++){
        if(C1_components_sizes[i] > 0){
            if(C1_components_sizes[i] > C1_components_sizes[max_size_root]){
                max_size_root = i;
            }
        }
    }
    
    // Call stratify if the largest component C1 has at least 4/5*|C| nodes
    if(C1_components_sizes[max_size_root] > c * 4/5){
        in_class_special<<< 1, n >>>(C1_components, max_size_root, C1);
        cudaDeviceSynchronize();
        my_uint128 newdelta = shl_my_uint128(delta, -1);
        stratify<<< 1, n >>>(numbering, C1, indptr, indices, newdelta.hi, newdelta.lo, n);
        cudaDeviceSynchronize();
    }

    cudaFree(adjacencies);
    cudaFree(level);
    cudaFree(in_component);
    cudaFree(component_size);
    cudaFree(C1);
    cudaFree(C1_components);
    cudaFree(C_A);
    cudaFree(C1_components_sizes);
    cudaFree(arr_odd);
    cudaFree(arr_even);
    cudaFree(sum);
    cudaFree(b_root);
    cudaFree(D);
    cudaFree(CuB);
    cudaFree(CuB_D);
    cudaFree(CuB_D_components);
    cudaFree(CuB_D_components_sum);
}

//Tries to break ties in numbering through stratification
__global__ void stratify(my_uint128 *numbering, float *roots, int *indptr, int *indices, unsigned long long int delta1, unsigned long long int delta2, int n)
{
    const int i = threadIdx.x;
    if(roots[i] != i) return;
    
    my_uint128 delta = llint_to_uint128(delta1, delta2);

    unsigned int pps_arr_size  = (n+1)*sizeof(float);

    float *unique, *unique_sum;
    cudaMalloc((void**)&unique, pps_arr_size);
    cudaMalloc((void**)&unique_sum, pps_arr_size);

    // Stratify must be executed only on nonsingleton class components
    am_unique<<< 1, n >>>(numbering, numbering[i], unique);
    cudaDeviceSynchronize();
    parallel_prefix(unique, unique_sum, n);
    cudaDeviceSynchronize();
    if(unique_sum[n]==1){
        cudaFree(unique);
        cudaFree(unique_sum);
        return;
    }

    float *is_richer_neighbor, *high_degree, *is_class_component, *neighbors_in_c;
    float *irn_sum, *hd_sum, *icc_sum, *nic_sum;

    cudaMalloc((void**)&is_richer_neighbor, pps_arr_size);
    cudaMalloc((void**)&high_degree, pps_arr_size);
    cudaMalloc((void**)&is_class_component, pps_arr_size);
    cudaMalloc((void**)&neighbors_in_c, pps_arr_size);
    cudaMalloc((void**)&irn_sum, pps_arr_size);
    cudaMalloc((void**)&hd_sum, pps_arr_size);
    cudaMalloc((void**)&icc_sum, pps_arr_size);
    cudaMalloc((void**)&nic_sum, pps_arr_size);


    in_class<<< 1, n >>>(roots, roots[i], is_class_component);
    cudaDeviceSynchronize();
    parallel_prefix(is_class_component, icc_sum, n);
    cudaDeviceSynchronize();
    
    richer_neighbors<<< 1, n >>>(numbering, roots, indptr, indices, roots[i], icc_sum[n], is_richer_neighbor, high_degree, neighbors_in_c);
    cudaDeviceSynchronize();
    parallel_prefix(high_degree, hd_sum, n);
    parallel_prefix(is_richer_neighbor, irn_sum, n);
    cudaDeviceSynchronize();
    
    // Based on the number and the degree (number of neighbors in C) of richer neighbors we call different types of stratify
    if(irn_sum[n] == 0){
        stratify_none(numbering, is_class_component, indptr, indices, delta, n, icc_sum[n]);
    }else{
        if(hd_sum[n] >= irn_sum[n]){
            stratify_high_degree(numbering, is_class_component, indptr, indices, delta, n, is_richer_neighbor, irn_sum[n], icc_sum[n]);
        }else{
            stratify_low_degree(numbering, is_class_component, indptr, indices, delta, n, is_richer_neighbor, icc_sum[n]);
        }
    }

    cudaFree(is_richer_neighbor);
    cudaFree(high_degree);
    cudaFree(is_class_component);
    cudaFree(neighbors_in_c);
    cudaFree(irn_sum);
    cudaFree(hd_sum);
    cudaFree(icc_sum);
    cudaFree(nic_sum);
    cudaFree(unique);
    cudaFree(unique_sum);
}

}

"""

cuda_module = DynamicSourceModule(cuda_code, include_dirs=[os.path.join(os.getcwd(), '..', 'lib')], no_extern_c=True)
stratify = cuda_module.get_function("stratify")
split_classes = cuda_module.get_function("get_class_components_global")

N = 256
DENSITY = 0.5

G = generateChordalGraph(N, DENSITY, debug=False)
# G = generateGraph(N, DENSITY)
Gcsr = nx.to_scipy_sparse_matrix(G)
numbering = np.zeros(N, dtype=np.complex128)

shifts = math.ceil(math.log(N, 5/4)) * 3
delta1 = np.uint64(0)
delta2 = np.uint64(0)
if shifts > 63:
    delta1 = np.uint64(2**(shifts - 64))
else:
    delta2 = np.uint64(2**(shifts))

extra_space = int(N / 16 + N / 16**2 + 1)
unique_numberings = np.unique(numbering)
start = time.time()
iterations = 0
#While the numbering is not one-to-one and delta is an integer
while len(unique_numberings) < len(numbering) and (delta1 >= 1 or delta2 >= 1):
    roots = np.arange(N, dtype=np.float32)
    #Get all the class components
    split_classes(cuda.In(numbering), cuda.In(Gcsr.indptr), cuda.In(Gcsr.indices), cuda.In(np.ones(N, dtype=np.float32)), np.int32(N), cuda.InOut(roots), block=(1, 1, 1), shared=8*(N+extra_space+10))
    #Call the stratify on each node
    #If a node is the root of a valid class component goes on with the stratification, otherwise it ends
    stratify(cuda.InOut(numbering), cuda.In(roots), cuda.In(Gcsr.indptr), cuda.In(Gcsr.indices), delta1, delta2, np.int32(N), block=(N, 1, 1), shared=8*(N+extra_space+10))
    delta1, delta2 = delta_divide(delta1, delta2)
    unique_numberings = np.unique(numbering)
    iterations += 1
print(numbering)
end = time.time()
if(len(unique_numberings) == len(numbering)):
    print("UNIQUE NUMBERING: "+str(end-start))
else:
    print("NOT UNIQUE NUMBERING: "+str(end-start))

start = time.time()
if(nx.is_chordal(G)):
    end = time.time()
    print("CHORDAL: "+str(end-start))
else:
    end = time.time()
    print("NOT CHORDAL: "+str(end-start))
