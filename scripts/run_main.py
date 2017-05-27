import networkx as nx
from peo_pycuda.chordal_gen import generateChordalGraph

N = 50
DENSITY = 0.5

G = generateChordalGraph(N, DENSITY)
Gcsr = nx.to_scipy_sparse_matrix(G)
print(Gcsr)