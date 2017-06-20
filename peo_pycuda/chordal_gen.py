import random
import math
import networkx as nx


#Generates a generic graph with given number of nodes and density
def generateGraph(N, DENSITY, debug=False):
    M = int((N * (N-1) / 2) * DENSITY)
    graph = []
    n = m = 0
    while n < N:
        graph.append(set())
        n += 1
    while m < M:
        n1 = random.randint(0, N-1)
        n2 = random.randint(0, N-1)
        if n1 != n2 and n1 not in graph[n2]:
            graph[n1].add(n2)
            graph[n2].add(n1)
            m += 1
    edgelist = []
    for i in range(len(graph)):
        for n in graph[i]:
            if n > i:
                edgelist.append((i, n))
    return nx.Graph(edgelist)


#Generates a chordal graph with given number of nodes and density
def generateChordalGraph(N, DENSITY, debug=False, type = 'mesh'):
    #Unfortunately DENSITY is not used yet
    M = int((N * (N-1) / 2) * DENSITY)
    graph = [set()]
    n = m = 0
    count1 = count2 = 0
    if type == 'clique-chain':
        graph = []
        k = int(N**2 - DENSITY * N **2 + DENSITY * N)//(N-2)
        clique_size = N//k
        for i in range(k):
            add_clique(graph, clique_size, i)
            n+=clique_size
    elif type == 'clique-add':
        graph = []
        clique_size = N//2
        add_clique(graph, clique_size, 0)
    while n < N-1 and type != 'clique-chain':
        if type == 'mesh':
            if random.random() < math.exp(-m * 3 / N):
                n1 = random.randint(0, len(graph)-1)
                graph.append(set([n1]))
                graph[n1].add(n+1)
                if debug:
                    count1 += 1
                    print("1. Added "+str(n)+" to "+str(n1))
                n += 1
                m += 1
            else:
                n1 = random.randint(0, len(graph)-1)
                while not graph[n1]:
                    n1 = random.randint(0, len(graph) - 1)
                n2 = random.sample(graph[n1], 1)[0]
                if debug:
                    count2 += 2
                    print("2. Added "+str(n1)+" and "+str(n2)+" to "+str(n))
                graph.append(set([n1, n2]))
                graph[n1].add(n+1)
                graph[n2].add(n+1)
                n += 1
                m += 2
        elif type == 'chain':
            n1 = random.randint(0, len(graph) - 1)
            graph.append(set([n1]))
            graph[n1].add(n + 1)
            if debug:
                count1 += 1
                print("1. Added " + str(n) + " to " + str(n1))
            n += 1
            m += 1
        elif type == 'clique-add':
            clique_nodes = random.sample(range(clique_size), random.randint(0,int(math.sqrt(N))))
            graph.append(set(clique_nodes))
            for node in clique_nodes:
                graph[node].add(len(graph)-1)
            n+=1
    if debug:
        print("Generated, 1: "+str(count1)+", 2: "+str(count2))
    edgelist = []
    for i in range(len(graph)):
        for n in graph[i]:
            if n > i:
                edgelist.append((i, n))
    return nx.Graph(edgelist)


#Exports a nx.Graph object in a specified file using CSR format
def exportGraphCsr(G, filename="graph.txt", path="../graphs"):
    destination = path + "/" + filename
    Gcsr = nx.to_scipy_sparse_matrix(G)
    out_file = open(destination, "w")
    out_file.write(str(len(Gcsr.indptr)) + " "+str(len(Gcsr.indices)) + " " + str(len(Gcsr.data)) + "\n")
    for i in Gcsr.indptr:
        out_file.write(str(i) + "\n")
    for i in Gcsr.indices:
        out_file.write(str(i) + "\n")
    for i in Gcsr.data:
        out_file.write(str(i) + "\n")
    out_file.close()

def add_clique(graph, clique_size, clique_num):
    first_node = clique_size*clique_num
    node_list = []
    for i in range(clique_size):
        node_list.append(set(range(first_node,first_node+clique_size)))
        node_list[i].remove(first_node+i)
    if clique_num>0:
        graph[-1].add(first_node)
        node_list[0].add(len(graph)-1)
    graph.extend(node_list)

