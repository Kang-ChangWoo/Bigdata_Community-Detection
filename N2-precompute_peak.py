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

from utils.utils import load_network, save_communities_to_file
from utils.detection import detect_communities_louvain, detect_communities_leiden


random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)


def main(args):
    graph_dir = os.path.join(args.root_path, args.dataset_path)
    temp_dir = os.path.join(args.root_path, f'{args.dataset_path}(temp)')

    os.makedirs(temp_dir, exist_ok=True)

    fpaths = os.listdir(graph_dir)

    if args.testset == 'real':
        dat_fpaths = [i for i in fpaths if not fnmatch.fnmatch(i, '*-community.dat')]
        dat_fpaths = [i for i in dat_fpaths if i.endswith('dat')]
        dat_fpaths = sorted(dat_fpaths)

        gt_fpaths = [i for i in fpaths if fnmatch.fnmatch(i, '*-community.dat')]
        gt_fpaths = [i for i in gt_fpaths if i.endswith('dat')]
        gt_fpaths = sorted(gt_fpaths)

    elif args.testset == 'TC1':
        dat_fpaths = [i for i in fpaths if not fnmatch.fnmatch(i, '*-c.dat')]
        dat_fpaths = [i for i in dat_fpaths if i.endswith('dat')]
        dat_fpaths = sorted(dat_fpaths)

        gt_fpaths = [i for i in fpaths if fnmatch.fnmatch(i, '*-c.dat')]
        gt_fpaths = [i for i in gt_fpaths if i.endswith('dat')]
        gt_fpaths = sorted(gt_fpaths)

    print("Saved to", os.path.join(graph_dir, 'label.txt'))

    for idx, fpath in tqdm(enumerate(dat_fpaths), desc=f' Extract nmi score from given graph.. ', total=len(dat_fpaths)):
        graph_fpath = os.path.join(graph_dir, fpath)
        gt_fpath = os.path.join(graph_dir, gt_fpaths[idx])
        G = load_network(gt_fpath)
        top_nmi = - np.inf
        top_res = - np.inf

        for i in tqdm(np.arange(0.5, 50.1, 0.1), total=len(np.arange(0.5, 50.1, 0.1))):
            detected_communities = detect_communities_louvain(G, i)

            saved_fpath = os.path.join(temp_dir, fpath).replace('.dat', '.cmty')

            save_communities_to_file(detected_communities, saved_fpath)

            detected_communities = load_ground_truth(saved_fpath)
            true_communities = load_ground_truth(gt_fpath)

            nmi_score = calculate_nmi(true_communities, detected_communities)

            if top_nmi < nmi_score:
                top_nmi = nmi_score
                top_res = i

            with open(os.path.join(temp_dir, fpath.replace('.dat', '.txt')), "a") as f:
                text = f"[{fpath}] iter,{str(idx).zfill(5)} resolution,{i:.6f} nmi,{nmi_score:.6f}\n"
                f.write(text)
                f.close()

            # print(f'Resolution: {i:.1f}, nmi: {nmi_score:.3f}')

        fname = f'label.txt'
        
        with open(os.path.join(graph_dir, fname), "a") as file:
            text = f"{fpath} {top_nmi:.6f} {top_res:.6f}\n"
            file.write(text)
            file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epoch', type=int, default=150, help='epoch')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset_path', type=str, default='real_trainset', help='random seed')
    parser.add_argument('--root_path', type=str, default='/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train', help='random seed')
    parser.add_argument('--testset', type=str, default='TC1', help='-')
    args = parser.parse_args()
    main(args)