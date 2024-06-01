import networkx as nx
import community as community_louvain
import random
import numpy as np
import argparse

from evaluation import eval, calculate_nmi, load_ground_truth
from sklearn.metrics.cluster import normalized_mutual_info_score

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Reading the Network Data
def load_network(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G

# Applying the Louvain Algorithm
def detect_communities_louvain(G, resolution_ = 1.0):
    partition = community_louvain.best_partition(G, resolution=resolution_)
    # Convert partition dictionary to list of lists for NMI calculation
    community_to_nodes = {}
    for node, community in partition.items():
        if community not in community_to_nodes:
            community_to_nodes[community] = []
        community_to_nodes[community].append(node)
    return list(community_to_nodes.values())


# Step 5: Save the result
def save_communities_to_file(communities, file_path):
    # Convert the list of lists into a dictionary with community as key and nodes as values
    community_dict = {}
    for community_id, nodes in enumerate(communities):
        for node in nodes:
            community_dict[node] = community_id

    # Sort the dictionary by community key (which are the node numbers here)
    sorted_community_items = sorted(community_dict.items())

    # Write to file, now ensuring nodes are listed in the sorted order of their community keys
    with open(file_path, 'w') as f:
        for node, community_id in sorted_community_items:
            #f.write(f"{node} {community_id}\n")
            f.write(str(node) + " " + str(community_id) + "\n")


parser = argparse.ArgumentParser(description='Detect communities in a network.')
parser.add_argument('--networkFile', '-n', type=str, help='The path of the network file.', default="./data/1-1.dat")
args = parser.parse_args()

community_file_path = args.networkFile.replace('.dat', '.cmty')

# for i in np.arange(0.1, 50.1, 0.1):
#     G = load_network(args.networkFile)

#     # Detect communities using Louvain method
#     detected_communities = detect_communities_louvain(G, i)

#     save_communities_to_file(detected_communities, community_file_path)

#     parser = argparse.ArgumentParser(description='Detect communities in a network.')
#     parser.add_argument('--networkFile', '-n', type=str, help='The path of the network file.', default="./data/1-1.dat")
#     args = parser.parse_args()

#     network_file_path = args.networkFile # network_file_path
#     print(f'Resolution: {i:.1f}', f'      evaluation: {eval(network_file_path)}')



fpath = ["../TC1/TC1-1/1-1.dat", "../TC1/TC1-2/1-2.dat", "../TC1/TC1-3/1-3.dat", "../TC1/TC1-4/1-4.dat", "../TC1/TC1-5/1-5.dat", "../TC1/TC1-6/1-6.dat", "../TC1/TC1-7/1-7.dat",
"../TC1/TC1-8/1-8.dat", "../TC1/TC1-9/1-9.dat", "../TC1/TC1-10/1-10.dat", "../TC2/tc11.dat", "../TC2/tc12.dat", "../TC2/tc13.dat", "../TC2/tc14.dat", "../TC2/tc15.dat"]

# degree print
from collections import Counter

for f_ in ["../TC1/TC1-1/1-1.dat"]:
    G = load_network(f_)
    degrees = [deg for node, deg in G.degree()]
    print(f_, Counter(degrees))
    print()
