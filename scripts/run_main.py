import networkx as nx
import numpy as np
import math
import collections

from peo_pycuda.chordal_gen import generateChordalGraph

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

def collect_components(components, vertices):
    ret = collections.defaultdict(list)
    for i, c in enumerate(components):
        ret[c].append(vertices[i])
    return sorted(ret.values(), key=lambda x: len(x), reverse=True)

cuda_code = """
#include <stdio.h>

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

"""

cuda_module = SourceModule(cuda_code)
stratify = cuda_module.get_function("stratify")
split_classes = cuda_module.get_function("split_classes")

N = 50
DENSITY = 0.5

G = generateChordalGraph(N, DENSITY)
Gcsr = nx.to_scipy_sparse_matrix(G)
numbering = np.zeros(N, dtype=np.float64)

delta = 8 ** math.ceil(math.log(N, 5/4))

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