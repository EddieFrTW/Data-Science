# program to find K-Cores of a graph 
from collections import defaultdict
import time
import sys
from func_timeout import func_set_timeout
K = 1200;
@func_set_timeout(30)
def task(in_file, out_file):
    #tStart = time.time()
    f = open(in_file, 'r')
    g = Graph(82168);
    for line in f.readlines():
        l = line.strip('\n')
        g.addEdge(int(l.split(' ')[0]), int(l.split(' ')[1]))
    #cnt_index = sorted(g.graph, key=lambda k: len(g.graph[k]))
    cnt_index = [x for x in g.graph]
    #vertices which have vDegree < K at the beginning
    matches = [x for x in cnt_index if len(g.graph[x]) < K]
    g.Find_K_core(cnt_index, K)    
    output_nodes = sorted(g.graph.keys())
    
    with open(out_file, 'w') as f:
        for item in output_nodes:
            f.write(str(item)+'\n')
    #tEnd = time.time()
    #print(tEnd-tStart)
class Graph: 
    
    def __init__(self,vertices): 
        self.V = vertices #No. of vertices 
		# default dictionary to store graph 
        self.graph= defaultdict(list) 
	# function to add an edge to undirected graph 
    
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
        self.graph[v].append(u)

    def Find_K_core(self, cnt_index, K):
        del_nodes = [x for x in cnt_index if len(self.graph[x]) < K];
        while del_nodes:
            for ind in del_nodes:
                for vertex in self.graph[ind]:
                    self.graph[vertex].remove(ind)
                self.graph.pop(ind) 
            cnt_index = [x for x in self.graph]
            del_nodes = [x for x in cnt_index if len(self.graph[x]) < K]
    

if __name__ == '__main__':
    if sys.argv[1]:
        task(sys.argv[1], sys.argv[2]);