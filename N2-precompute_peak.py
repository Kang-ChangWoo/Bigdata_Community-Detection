import os, fnmatch, random, argparse

import networkx as nx
import community as community_louvain
import numpy as np

from sklearn.metrics.cluster import normalized_mutual_info_score
from natsort import natsorted
from tqdm import tqdm

from utils.utils import load_network, save_communities_to_file, load_ground_truth
from utils.utils import extract_meta_info, calculate_nmi, evaluate_nmi

from utils.detection import detect_communities_louvain, detect_communities_leiden

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    graph_dir = os.path.join(args.root_path, args.dataset_path)
    temp_dir = os.path.join(args.root_path, f'{args.dataset_path}(temp)')

    os.makedirs(temp_dir, exist_ok=True)

    fpaths = os.listdir(graph_dir)
    gpaths = os.listdir(graph_dir)

    if args.testset == 'real':
        dat_fpaths = [i for i in fpaths if not fnmatch.fnmatch(i, '*-community.dat')]
        dat_fpaths = [i for i in dat_fpaths if i.endswith('dat')]
        dat_fpaths = natsorted(dat_fpaths)

        gt_fpaths = [i for i in fpaths if fnmatch.fnmatch(i, '*-community.dat')]
        gt_fpaths = [i for i in gt_fpaths if i.endswith('dat')]
        gt_fpaths = natsorted(gt_fpaths)

    elif args.testset == 'TC1':
        dat_fpaths = [i for i in fpaths if not fnmatch.fnmatch(i, '*-c.dat')]
        dat_fpaths = [i for i in dat_fpaths if i.endswith('dat')]
        dat_fpaths = natsorted(dat_fpaths)

        gt_fpaths = [i for i in fpaths if fnmatch.fnmatch(i, '*-*-c.dat')]
        gt_fpaths = [i for i in gt_fpaths if i.endswith('dat')]
        gt_fpaths = natsorted(gt_fpaths)

    print("Saved to", os.path.join(graph_dir, 'label.txt'))

    for idx, fpath in tqdm(enumerate(dat_fpaths), desc=f' Extract nmi score from given graph.. ', total=len(dat_fpaths)):
        graph_fpath = os.path.join(graph_dir, fpath)
        gt_fpath = os.path.join(graph_dir, gt_fpaths[idx])
        G = load_network(gt_fpath)
        top_nmi = - np.inf
        top_res = - np.inf

        for i in tqdm(np.arange(0.5, 50.1, 0.1), total=len(np.arange(0.5, 50.1, 0.1))):
            pred_communities = detect_communities_leiden(G, resolution=i)
            saved_fpath = os.path.join(temp_dir, fpath).replace('.dat', '.cmty')

            save_communities_to_file(pred_communities, saved_fpath)

            pred_communities = load_ground_truth(saved_fpath)
            true_communities = load_ground_truth(gt_fpath)

            nmi_score = calculate_nmi(true_communities, pred_communities)

            if top_nmi < nmi_score:
                top_nmi = nmi_score
                top_res = i

            with open(os.path.join(temp_dir, fpath.replace('.dat', '.txt')), "a") as f:
                text = f"[{fpath}] iter,{str(idx).zfill(5)} resolution,{i:.6f} nmi,{nmi_score:.6f}\n"
                f.write(text)
                f.close()

        fname = f'label.txt'
        
        with open(os.path.join(graph_dir, fname), "a") as file:
            text = f"{fpath} {top_nmi:.6f} {top_res:.6f}\n"
            file.write(text)
            file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset_path', type=str, default='graph_dataset_v2', help='random seed')
    parser.add_argument('--root_path', type=str, default='/root/storage/implementation/Lecture-BDB_proj/Bigdata_Community-Detection/data/train', help='random seed')
    parser.add_argument('--testset', type=str, default='TC1', help='-')
    args = parser.parse_args()
    main(args)