from peo_pycuda.chordal_gen import generateChordalGraph
import numpy as np
from scipy import sparse
import collections
import networkx as nx



def collect_components(components, vertices):
    ret = collections.defaultdict(list)
    for i, c in enumerate(components):
        ret[c].append(vertices[i])
    return sorted(ret.values(), key=lambda x: len(x), reverse=True)


g = generateChordalGraph(500, 0.4, False)

slice = np.random.random_integers(0, 500, 200)

vertices, edges = g.nodes(), nx.to_scipy_sparse_matrix(g)[:,slice][slice,:]
n, components = sparse.csgraph.connected_components(edges, directed=False)
components = collect_components(components,vertices)
print(len(components))