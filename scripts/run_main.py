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

/*
#ifndef _SCAN_BEST_KERNEL_H_
#define _SCAN_BEST_KERNEL_H_

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#ifdef CHECK_BANK_CONFLICTS
#define TEMP(index)   CUT_BANK_CHECKER(temp, index)
#else
#define TEMP(index)   temp[index]
#endif

///////////////////////////////////////////////////////////////////////////////
// Work-efficient compute implementation of scan, one thread per 2 elements
// Work-efficient: O(log(n)) steps, and O(n) adds.
// Also shared storage efficient: Uses n + n/NUM_BANKS shared memory -- no ping-ponging
// Also avoids most bank conflicts using single-element offsets every NUM_BANKS elements.
//
// In addition, If ZERO_BANK_CONFLICTS is defined, uses 
//     n + n/NUM_BANKS + n/(NUM_BANKS*NUM_BANKS) 
// shared memory. If ZERO_BANK_CONFLICTS is defined, avoids ALL bank conflicts using 
// single-element offsets every NUM_BANKS elements, plus additional single-element offsets 
// after every NUM_BANKS^2 elements.
//
// Uses a balanced tree type algorithm.  See Blelloch, 1990 "Prefix Sums 
// and Their Applications", or Prins and Chatterjee PRAM course notes:
// http://www.cs.unc.edu/~prins/Classes/203/Handouts/pram.pdf
// 
// This work-efficient version is based on the algorithm presented in Guy Blelloch's
// Excellent paper "Prefix sums and their applications".
// http://www-2.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html
//
// Pro: Work Efficient, very few bank conflicts (or zero if ZERO_BANK_CONFLICTS is defined)
// Con: More instructions to compute bank-conflict-free shared memory addressing,
// and slightly more shared memory storage used.
//
// @param g_odata  output data in global memory
// @param g_idata  input data in global memory
// @param n        input number of elements to scan from input data
__global__ void scan_best(double *g_odata, double *g_idata, int n)
{
    // Dynamically allocated shared memory for scan kernels
    extern  __shared__  double temp[];

    int thid = threadIdx.x;

    int ai = thid;
    int bi = thid + (n/2);

    // compute spacing to avoid bank conflicts
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Cache the computational window in shared memory
    TEMP(ai + bankOffsetA) = g_idata[ai]; 
    TEMP(bi + bankOffsetB) = g_idata[bi];
    printf("QUI 1\\n");

    int offset = 1;

    // build the sum in place up the tree
    for (int d = n>>1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            TEMP(bi) += TEMP(ai);
        }

        offset *= 2;
    }
    printf("QUI 2\\n");

    // scan back down the tree

    // clear the last element
    if (thid == 0)
    {
        int index = n - 1;
        index += CONFLICT_FREE_OFFSET(index);
        TEMP(index) = 0;
    }   
    printf("QUI 3\\n");

    // traverse down the tree building the scan in place
    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();

        if (thid < d)
        {
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;

            printf("QUI 3.1: %d %d %d\\n",offset,ai,bi);
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            printf("QUI 3.2: %d %d\\n",ai,bi);
            double t  = TEMP(ai);
            TEMP(ai) = TEMP(bi);
            TEMP(bi) += t;
            printf("QUI 3.3: %d %d\\n",ai,bi);
        }
    }
    printf("QUI 4\\n");

    __syncthreads();

    // write results to global memory
    g_odata[ai] = TEMP(ai + bankOffsetA);
    g_odata[bi] = TEMP(bi + bankOffsetB);
    
    printf("QUI 5\\n");
}
#endif // #ifndef _SCAN_BEST_KERNEL_H_*/

extern "C" {
__global__ void parallel_prefix(float *d_idata, float *d_odata, int num_elements)
{
    
    float** g_scanBlockSums;
    unsigned int g_numEltsAllocated = num_elements;
    unsigned int g_numLevelsAllocated = 0;

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
    g_numLevelsAllocated = level;
    
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

    prescanArrayRecursive(d_odata, d_idata, num_elements, 0, g_scanBlockSums, g_numEltsAllocated, g_numLevelsAllocated);
    
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

__global__ void stratify(double *numbering, long long int *roots, int *indptr, int *indices, double delta)
{
    const int i = threadIdx.x;
    if(roots[i] != i) return;
    //TODO
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

tmp = np.zeros(N, dtype=np.float64)
prescan(cuda.In(np.arange(N, dtype=np.float32)), cuda.Out(tmp), np.int32(N), block=(1,1,1), shared=8*(N+extra_space+10))
print(np.array(tmp, dtype=np.int))

unique_numberings, unique_numberings_idx = np.unique(numbering, return_inverse=True)
while len(unique_numberings) < len(numbering) and delta >= 1:
    changes = 1
    roots = np.arange(N, dtype=np.int64)
    while np.sum(changes) > 0:
        changes = np.zeros(N, dtype=np.int)
        split_classes(cuda.In(numbering), cuda.In(Gcsr.indptr), cuda.In(Gcsr.indices), cuda.InOut(roots), cuda.InOut(changes), block=(N, 1, 1))
    stratify(cuda.InOut(numbering), cuda.In(roots), cuda.In(Gcsr.indptr), cuda.In(Gcsr.indices), np.float64(delta), block=(N, 1, 1))
    delta /= 8
    unique_numberings, unique_numberings_idx = np.unique(numbering, return_inverse=True)