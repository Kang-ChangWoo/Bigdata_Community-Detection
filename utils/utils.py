import os, glob, argparse, fnmatch

import networkx
from natsort import natsorted
import torch
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score

# Input: Graph / device inform 
# Output: Meta information (tensor type)
# Used in: Network initialization
def extract_meta_info(G, device='cpu'):
    meta_info = {}

    meta_info['n_nodes'] = G.number_of_nodes()
    meta_info['n_edges'] = G.number_of_edges()
    meta_info['avg_deg'] = sum(dict(G.degree()).values()) / meta_info['n_nodes']
    meta_info['density'] = networkx.density(G)
    meta_info['n_connected_comp'] = networkx.number_connected_components(G)
    meta_info['avg_cluster'] = networkx.average_clustering(G)
    meta_info['transitivity'] = networkx.transitivity(G)

    meta_vec = torch.tensor([[meta_info['n_nodes'],
                            meta_info['n_edges'],
                            meta_info['avg_deg'],
                            meta_info['density'],
                            meta_info['n_connected_comp'],
                            meta_info['avg_cluster'],
                            meta_info['transitivity'],]],
                            dtype=torch.float32)
                            
    return meta_vec.to(device)


# Input: file path
# Output: loaded graph (network x graph type)
# Used in: Dataset load (graph)
def load_network(file_path):
    G = networkx.Graph()
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                G.add_edge(int(parts[0]), int(parts[1]))
    return G

# Input: file_path
# Output: loaded community (node)
# Used in: Dataset load (graph)
def load_ground_truth(file_path):
    node_to_community = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                node, community = int(parts[0]), int(parts[1])
                node_to_community[node] = community
    # Convert to list of lists for compatibility with NMI calculation
    community_to_nodes = {}
    for node, community in node_to_community.items():
        if community not in community_to_nodes:
            community_to_nodes[community] = []
        community_to_nodes[community].append(node)
    return list(community_to_nodes.values())


# Input: G.T. / Pred communities
# Output: Normalized score
# Used in: evaluation
def calculate_nmi(true_communities, detected_communities):
    # Flatten the lists and create label vectors
    true_labels = {}
    for i, community in enumerate(true_communities):
        for node in community:
            true_labels[node] = i
    detected_labels = {}
    for i, community in enumerate(detected_communities):
        for node in community:
            detected_labels[node] = i

    # Ensure the labels are in the same order for both true and detected
    nodes = natsorted(set(true_labels) | set(detected_labels))
    true_labels_vector = [true_labels[node] for node in nodes]
    detected_labels_vector = [detected_labels.get(node, -1) for node in nodes]

    return normalized_mutual_info_score(true_labels_vector, detected_labels_vector)


# Input: communities(list?) / file path
# Output: saved community
# Used in: save computed communities
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


def evaluate_nmi(network_file_path):
    # Replace or append file extensions as necessary to construct paths
    community_file_path = network_file_path.replace('.dat', '.cmty')
    ground_truth_file_path = network_file_path.replace('.dat', '-c.dat')

    detected_communities = load_ground_truth(community_file_path)
    true_communities = load_ground_truth(ground_truth_file_path)
    nmi_score = calculate_nmi(true_communities, detected_communities)
    return nmi_score