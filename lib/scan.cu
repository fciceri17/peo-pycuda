/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */
#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <scanLargeArray_kernel.cu>
#include <assert.h>

inline bool 
__host__ __device__ isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}

inline int 
__host__ __device__ floorPow2(int n)
{
#ifdef WIN32
    // method 2
    return 1 << (int)logb((float)n);
#else
    // method 1
    // float nf = (float)n;
    // return 1 << (((*(int*)&nf) >> 23) - 127); 
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

#define BLOCK_SIZE 256


__host__ __device__ void prescanArrayRecursive(float *outArray, 
                           const float *inArray, 
                           int numElements, 
                           int level,
                           float **g_scanBlockSums)
{
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks = 
        max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;

    // if this is a non-power-of-2 array, the last block will be non-full
    // compute the smallest power of 2 able to compute its scan.
    unsigned int numEltsLastBlock = 
        numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;
    
    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;

        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);    
        
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = 
            sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = 
        sizeof(float) * (numEltsPerBlock + extraSpace);

    // setup execution parameters
    // if NP2, we process the last block separately
    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1); 
    dim3  threads(numThreads, 1, 1);


    // execute the scan
    if (numBlocks > 1)
    {
        prescan<true, false><<< grid, threads, sharedMemSize >>>(outArray, 
                                                                 inArray, 
                                                                 g_scanBlockSums[level],
                                                                 numThreads * 2, 0, 0);
        if (np2LastBlock)
        {
            prescan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>>
                (outArray, inArray, g_scanBlockSums[level], numEltsLastBlock, 
                 numBlocks - 1, numElements - numEltsLastBlock);
        }

        // After scanning all the sub-blocks, we are mostly done.  But now we 
        // need to take all of the last values of the sub-blocks and scan those.  
        // This will give us a new value that must be sdded to each block to 
        // get the final results.
        // recursive (CPU) call
        prescanArrayRecursive(g_scanBlockSums[level], 
                              g_scanBlockSums[level], 
                              numBlocks, 
                              level+1, g_scanBlockSums);

        uniformAdd<<< grid, threads >>>(outArray, 
                                        g_scanBlockSums[level], 
                                        numElements - numEltsLastBlock, 
                                        0, 0);
        if (np2LastBlock)
        {
            uniformAdd<<< 1, numThreadsLastBlock >>>(outArray, 
                                                     g_scanBlockSums[level], 
                                                     numEltsLastBlock, 
                                                     numBlocks - 1, 
                                                     numElements - numEltsLastBlock);
        }
    }
    else if (isPowerOfTwo(numElements))
    {
        prescan<false, false><<< grid, threads, sharedMemSize >>>(outArray, inArray,
                                                                  0, numThreads * 2, 0, 0);
    }
    else
    {
         prescan<false, true><<< grid, threads, sharedMemSize >>>(outArray, inArray, 
                                                                  0, numElements, 0, 0);
    }
}


#endif // _PRESCAN_CU_
