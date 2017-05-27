import networkx as nx
import numpy as np
import math

from peo_pycuda.chordal_gen import generateChordalGraph

N = 50
DENSITY = 0.5

G = generateChordalGraph(N, DENSITY)
Gcsr = nx.to_scipy_sparse_matrix(G)
numbering = np.zeros(N, dtype=np.int)
delta = 8 ** math.ceil(math.log(N, 5/4))
print(Gcsr)