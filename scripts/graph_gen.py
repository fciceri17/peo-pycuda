import peo_pycuda.chordal_gen as cg

for i in range(10):
    n = (i+1) * 10
    G = cg.generateGraph(n, 0.5)
    cg.exportGraphCsr(G, filename="graph_"+str(n)+".txt")