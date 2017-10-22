
import random
import networkx as nx
from Autoencoder_st import *


# read graph.txt
# store lines of the graph in a graph_dict
# since it's undirected graph --- for (node1, node2), graph_dict[node1] = [..., node2], graph_dict[node2] = [..., node1]

f_graph = open("newman.txt", "r")
G = nx.Graph()

for line in f_graph:
    l= line.split(" ") # read node1 node2
    node1 = int(l[0])
    node2 = int(l[1])
    G.add_edge(node1, node2)
    # if node1 in graph_dict:
    #     graph_dict[node1].append(node2)
    # else:
    #     graph_dict[node1] = [node2]
    # if node2 in graph_dict:
    #     graph_dict[node2].append(node1)
    # else:
    #     graph_dict[node2] = [node1]
f_graph.close()
    

# sim_list = []
# for x in range(2000):
sim = Simulation(G)
num_input = np.array(sim).shape[1]
# print(num_input)
#     sim_list.append(Re_sim)

model(G, 200, 200, sim)






