import networkx as nx
import community as community_louvain
import random
import numpy as np
import argparse

from evaluation import eval, calculate_nmi, load_ground_truth
from sklearn.metrics.cluster import normalized_mutual_info_score

import os
import fnmatch
from tqdm import tqdm


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


import os
import shutil

def move_and_rename_dat_files(path):
    # 최상위 디렉토리의 서브디렉토리 목록 가져오기
    subdirectories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for subdirectory in subdirectories:
        sub_path = os.path.join(path, subdirectory)
        # 서브디렉토리 내의 항목 목록 가져오기
        items = os.listdir(sub_path)

        for item in items:
            item_path = os.path.join(sub_path, item)
            # .dat 파일인 경우 새로운 이름으로 최상위 디렉토리로 이동
            if item.endswith('.dat'):
                new_name = f"{subdirectory}-{item}"
                new_path = os.path.join(path, new_name)
                shutil.move(item_path, new_path)
                print(f"Moved {item} to {new_path}")
        
        # 서브디렉토리 및 그 안의 모든 파일 삭제
        shutil.rmtree(sub_path)
        print(f"Deleted directory {sub_path}")

def main(args):
    if args.isit_hierarchy == 1:
        move_and_rename_dat_files(args.dataset_path)







    # graph_dir = os.path.join(args.dataset_path, 'graph_dataset_v2')
    # temp_dir = os.path.join(args.dataset_path, 'graph_dataset_v2(temp)')

    # os.makedirs(temp_dir, exist_ok=True)

    # fpaths = os.listdir(graph_dir)
    # dat_fpaths = [i for i in fpaths if not fnmatch.fnmatch(i, '*-c.dat')]
    # dat_fpaths = sorted(dat_fpaths)

    # gt_fpaths = [i for i in fpaths if fnmatch.fnmatch(i, '*-c.dat')]
    # gt_fpaths = sorted(gt_fpaths)

    # for idx, fpath in tqdm(enumerate(dat_fpaths), desc=f' Extract nmi score from given graph.. ', total=len(dat_fpaths)):
    #     graph_fpath = os.path.join(graph_dir, fpath)
    #     gt_fpath = os.path.join(graph_dir, gt_fpaths[idx])
    #     G = load_network(gt_fpath)
    #     top_nmi = - np.inf
    #     top_res = - np.inf

    #     for i in tqdm(np.arange(0.5, 50.1, 0.1), total=len(np.arange(0.1, 50.1, 0.1))):
    #         detected_communities = detect_communities_louvain(G, i)

    #         saved_fpath = os.path.join(temp_dir, fpath).replace('.dat', '.cmty')

    #         save_communities_to_file(detected_communities, saved_fpath)

    #         detected_communities = load_ground_truth(saved_fpath)
    #         true_communities = load_ground_truth(gt_fpath)

    #         nmi_score = calculate_nmi(true_communities, detected_communities)

    #         if top_nmi < nmi_score:
    #             top_nmi = nmi_score
    #             top_res = i

    #         with open(os.path.join(temp_dir, fpath.replace('.dat', '.txt')), "a") as f:
    #             text = f"[{fpath}] iter,{str(idx).zfill(5)} resolution,{i:.6f} nmi,{nmi_score:.6f}\n"
    #             f.write(text)
    #             f.close()
    #             #11

    #         # print(f'Resolution: {i:.1f}, nmi: {nmi_score:.3f}')

    #     fname = f'label.txt'
    #     with open(os.path.join(graph_dir, fname), "a") as file:
    #         text = f"{fpath} {top_nmi:.6f} {top_res:.6f}\n"
    #         file.write(text)
    #         file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', type=str, default='/root/storage/implementation/Lecture-BDB_proj/data/test/real-world_dataset', help='-')
    parser.add_argument('--isit_hierarchy', type=int, default=0, help='-')
    args = parser.parse_args()
    main(args)