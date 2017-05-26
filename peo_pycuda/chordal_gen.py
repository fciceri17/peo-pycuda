import random
import math
import networkx as nx


def generateChordalGraph(N, DENSITY, debug=False):
    M = int((N * (N-1) / 2) * DENSITY)
    graph = [set()]
    n = m = 0
    count1 = count2 = 0
    while n < N:
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
    if debug:
        print("Generated, 1: "+str(count1)+", 2: "+str(count2))
    edgelist = []
    for i in range(len(graph)):
        for n in graph[i]:
            if n > i:
                edgelist.append((i, n))
    return nx.Graph(edgelist)

g = generateChordalGraph(500, 0.5, True)
print(nx.is_chordal(g))