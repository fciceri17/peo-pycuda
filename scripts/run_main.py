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

extern "C" {
__host__ __device__ void parallel_prefix(float *d_idata, float *d_odata, int num_elements)
{

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

__global__ void split_classes(double *numbering, int *indptr, int *indices, long long int *roots, int *changes)
{
    const int i = threadIdx.x;
    int min = roots[i];

    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(numbering[i] == numbering[indices[j]] && roots[indices[j]] < min){
            min = roots[indices[j]];
        }
    }

    if(min != roots[i]){
        roots[i] = min;
        changes[i] += 1;
    }
}

__global__ void spanning_tree_depth(int *indptr, int *indices, int *level, int *in_component, int *neighbors, int curr_level)
{
    const int i = threadIdx.x;
    int curr_node = neighbors[i];
    if(level[curr_node]>0 || in_component[curr_node]==0)
        return;
    level[curr_node] = curr_level;

    int j = indptr[curr_node];
    int num_neighbors = indptr[curr_node+1] - indptr[curr_node];
    if(num_neighbors>0){
        __syncthreads();
        spanning_tree_depth<<< 1, num_neighbors >>>(indptr, indices, level, in_component, indices+j*sizeof(int), curr_level+1);
    }
}


//outputs level of depth forming a spanning tree for a given root in component. the level-node index pair gives a unique
//depth ordering for each node in the component

__global__ void spanning_tree_numbering(int *indptr, int *indices, int *in_component, int *level, int root, int len)
{

    level[root]=1;
    int j = indptr[root];
    int num_neighbors = indptr[root+1] - indptr[root];
    spanning_tree_depth<<< 1, num_neighbors >>>(indptr, indices, level, indices+j*sizeof(int), in_component, 2);
    cudaDeviceSynchronize();


}

__global__ void richer_neighbors(double *numbering, long long int *roots, int *indptr, int *indices, int root, float c, float *is_richer_neighbor, float *high_degree, float *neighbors_in_c)
{
    const int i = threadIdx.x;
    is_richer_neighbor[i] = 0;
    high_degree[i] = 0;
    neighbors_in_c[i] = 0;
    if(roots[i] == c) return;

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

__global__ void in_class(double *numbering, long long int *roots, int *indptr, int *indices, int c, float *is_class_component)
{
    const int i = threadIdx.x;
    is_class_component[i] = 0;
    if(roots[i] == c) is_class_component[i] = 1;
}

__device__ void stratify_none(double *numbering, long long int *roots, int *indptr, int *indices, double delta, int n, float c, long long int root)
{
    float *D, *C_D;

    unsigned int pps_arr_size  = n*sizeof(float);
    cudaMalloc((void**)&D, pps_arr_size);
    cudaMalloc((void**)&C_D, pps_arr_size);

    stratify_none_getD<<< 1, n >>>(numbering, roots, indptr, indices, n, c, root, D);
    cudaDeviceSynchronize();
    stratify_none_getC_D<<< 1, n >>>(numbering, roots, indptr, indices, n, D, root, C_D);
    cudaDeviceSynchronize();
}

__device__ void stratify_high_degree(double *numbering, long long int *roots, int *indptr, int *indices, double delta, int n, float *is_richer_neighbor)
{

}

__device__ void stratify_low_degree(double *numbering, long long int *roots, int *indptr, int *indices, double delta, int n, float *is_richer_neighbor)
{

}

__global__ void stratify(double *numbering, long long int *roots, int *indptr, int *indices, double delta, int n)
{
    const int i = threadIdx.x;
    if(roots[i] != i) return;

    float *is_richer_neighbor, *high_degree, *is_class_component, *neighbors_in_c;
    float *irn_sum, *hd_sum, *icc_sum, *nic_sum;

    unsigned int pps_arr_size  = n*sizeof(float);
    cudaMalloc((void**)&is_richer_neighbor, pps_arr_size);
    cudaMalloc((void**)&high_degree, pps_arr_size);
    cudaMalloc((void**)&is_class_component, pps_arr_size);
    cudaMalloc((void**)&neighbors_in_c, pps_arr_size);
    cudaMalloc((void**)&irn_sum, pps_arr_size);
    cudaMalloc((void**)&hd_sum, pps_arr_size);
    cudaMalloc((void**)&icc_sum, pps_arr_size);
    cudaMalloc((void**)&nic_sum, pps_arr_size);


    in_class<<< 1, n >>>(numbering, roots, indptr, indices, numbering[i], is_class_component);
    cudaDeviceSynchronize();
    parallel_prefix(is_class_component, icc_sum, n);
    cudaDeviceSynchronize();

    richer_neighbors<<< 1, n >>>(numbering, roots, indptr, indices, roots[i], icc_sum[n-1], is_richer_neighbor, high_degree, neighbors_in_c);
    cudaDeviceSynchronize();
    parallel_prefix(is_richer_neighbor, irn_sum, n);
    cudaDeviceSynchronize();
    if(irn_sum[n-1] == 0)
        stratify_none(numbering, roots, indptr, indices, delta, n, icc_sum[n-1], i);
    else{
        parallel_prefix(high_degree, hd_sum, n);
        cudaDeviceSynchronize();
        if(hd_sum[n-1] == irn_sum[n-1])
            stratify_high_degree(numbering, roots, indptr, indices, delta, n, is_richer_neighbor);
        else
            stratify_low_degree(numbering, roots, indptr, indices, delta, n, is_richer_neighbor);
    }


}

}

"""

cuda_module = DynamicSourceModule(cuda_code, include_dirs=[os.path.join(os.getcwd(), '..', 'lib')], no_extern_c=True)
stratify = cuda_module.get_function("stratify")
split_classes = cuda_module.get_function("split_classes")

N = 64
DENSITY = 0.5

G = generateChordalGraph(N, DENSITY)
Gcsr = nx.to_scipy_sparse_matrix(G)
numbering = np.zeros(N, dtype=np.float32)

delta = 8 ** math.ceil(math.log(N, 5/4))

extra_space = int(N / 16 + N / 16**2 + 1)

unique_numberings, unique_numberings_idx = np.unique(numbering, return_inverse=True)
while len(unique_numberings) < len(numbering) and delta >= 1:
    changes = 1
    roots = np.arange(N, dtype=np.int64)
    while np.sum(changes) > 0:
        changes = np.zeros(N, dtype=np.int)
        split_classes(cuda.In(numbering), cuda.In(Gcsr.indptr), cuda.In(Gcsr.indices), cuda.InOut(roots), cuda.InOut(changes), block=(N, 1, 1))
    stratify(cuda.InOut(numbering), cuda.In(roots), cuda.In(Gcsr.indptr), cuda.In(Gcsr.indices), np.float64(delta), np.int32(N), block=(N, 1, 1), shared=8*(N+extra_space+10))
    delta /= 8
    unique_numberings, unique_numberings_idx = np.unique(numbering, return_inverse=True)