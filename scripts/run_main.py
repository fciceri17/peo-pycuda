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

extern "C" {
__global__ void parallel_prefix(float *d_idata, float *d_odata, int num_elements)
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

__global__ void richer_neighbors(double *numbering, long long int *roots, int *indptr, int *indices, int root, int c, int *is_richer_neighbor, int *high_degree)
{
    const int i = threadIdx.x;
    is_richer_neighbor[i] = 0;
    high_degree[i] = 0;
    if(roots[i] == c) return;

    int neighbors_in_c = 0;
    for(int j = indptr[i]; j < indptr[i+1]; j++){
        if(numbering[i] > numbering[indices[j]] && roots[indices[j]] == root){
            is_richer_neighbor[i] = 1;
            neighbors_in_c += 1;
        }
    }

    if(neighbors_in_c >= 2 / 5 * c){
        high_degree[i] = 1;
    }
}

__global__ void in_class(double *numbering, long long int *roots, int *indptr, int *indices, int c, int *is_class_component)
{
    const int i = threadIdx.x;
    if(roots[i] == c) is_class_component[i] = 1;
}


__global__ void stratify_none(double *numbering, long long int *roots, int *indptr, int *indices, double delta, int n)
{

}

__global__ void stratify_high_degree(double *numbering, long long int *roots, int *indptr, int *indices, double delta, int n)
{

}

__global__ void stratify_low_degree(double *numbering, long long int *roots, int *indptr, int *indices, double delta, int n)
{

}

__global__ void stratify(double *numbering, long long int *roots, int *indptr, int *indices, double delta, int n)
{
    const int i = threadIdx.x;
    if(roots[i] != i) return;

    float *is_richer_neighbor, *high_degree, *is_class_component;
    float *irn_sum, *hd_sum, *icc_sum;

    unsigned int pps_arr_size  = n*sizeof(float)
    cudaMalloc(&is_richer_neighbor, pps_arr_size)
    cudaMalloc(&high_degree, pps_arr_size)
    cudaMalloc(&is_class_component, pps_arr_size)
    cudaMalloc(&irn_sum, pps_arr_size)
    cudaMalloc(&hd_sum, pps_arr_size)
    cudaMalloc(&icc_sum, pps_arr_size)


    in_class<<< 1, n >>>(numbering, roots, indptr, indices, numbering[i], is_class_component);
    parallel_prefix(is_class_copmponent, icc_sum, n);

    richer_neighbors<<< 1, n >>>(numbering, roots, indptr, indices, roots[i], icc_sum[n-1], is_richer_neighbor, high_degree);
    parallel_prefix(is_richer_neighbor, irn_sum, n);
    if(irn_sum[n-1] == 0)
        stratify_none(numbering, roots, indptr, indices, delta, n);
    else{
        parallel_prefix(high_degree, hd_sum, n);
        if(hd_sum[n-1] == irn_sum[n-1])
            stratify_high_degree(numbering, roots, indptr, indices, delta, n);
        else
            stratify_low_degree(numerbing, roots, indptr, indices, delta, n);
    }


}

}

"""

cuda_module = DynamicSourceModule(cuda_code, include_dirs=[os.path.join(os.getcwd(), '..', 'lib')], no_extern_c=True)
stratify = cuda_module.get_function("stratify")
split_classes = cuda_module.get_function("split_classes")
prescan = cuda_module.get_function("parallel_prefix")

N = 64
DENSITY = 0.5

G = generateChordalGraph(N, DENSITY)
Gcsr = nx.to_scipy_sparse_matrix(G)
numbering = np.zeros(N, dtype=np.float32)

delta = 8 ** math.ceil(math.log(N, 5/4))

extra_space = int(N / 16 + N / 16**2 + 1)

tmp = np.zeros(N, dtype=np.float32)
prescan(cuda.In(np.arange(N, dtype=np.float32)), cuda.Out(tmp), np.int32(N), block=(1,1,1), shared=8*(N+extra_space+10))
print(np.array(tmp, dtype=np.int))

unique_numberings, unique_numberings_idx = np.unique(numbering, return_inverse=True)
while len(unique_numberings) < len(numbering) and delta >= 1:
    changes = 1
    roots = np.arange(N, dtype=np.int64)
    while np.sum(changes) > 0:
        changes = np.zeros(N, dtype=np.int)
        split_classes(cuda.In(numbering), cuda.In(Gcsr.indptr), cuda.In(Gcsr.indices), cuda.InOut(roots), cuda.InOut(changes), block=(N, 1, 1))
    stratify(cuda.InOut(numbering), cuda.In(roots), cuda.In(Gcsr.indptr), cuda.In(Gcsr.indices), np.float64(delta), np.int32(N), block=(N, 1, 1))
    delta /= 8
    unique_numberings, unique_numberings_idx = np.unique(numbering, return_inverse=True)