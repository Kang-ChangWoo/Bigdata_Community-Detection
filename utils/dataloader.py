import os, glob, argparse, fnmatch

from natsort import natsorted
import networkx
import torch
from torch_geometric.utils import from_networkx
import numpy as np

from utils.utils import load_network, load_ground_truth

# Input: file_path / device inform
# Output: dataset (obj.)
# Used in: Dataset initialization
class train_graph_loader(torch.utils.data.Dataset):
    def __init__(self, file_path, device='cpu'):
        self.file_path = file_path
        
        self.graphs = glob.glob(os.path.join(file_path, '*-*.dat' ))
        self.graphs = [i for i in self.graphs if not fnmatch.fnmatch(i, '*-c.dat')]
        self.graphs = [i for i in self.graphs if i.endswith('dat')]
        self.graphs = natsorted(self.graphs)

        self.community_labels = glob.glob(os.path.join(file_path, '*-*-c.dat' ))
        self.community_labels = [i for i in self.community_labels if i.endswith('dat')]
        self.community_labels = natsorted(self.community_labels)

        self.device = device

        self.peak_resolutions = np.loadtxt(os.path.join(file_path,'label.txt'), dtype=str)

    def __getitem__(self, idx):
        G = load_network(self.graphs[idx])
        torch_G = from_networkx(G).to(self.device)
        torch_G.x = torch.ones((torch_G.num_nodes, 1), dtype=torch.float).to(self.device)

        community_label = self.community_labels[idx]
        _, nmi_score, peak_resolution = self.peak_resolutions[idx]

        nmi_score = torch.tensor([float(nmi_score)]).to(self.device)
        peak_resolution = torch.tensor([float(peak_resolution)]).to(self.device)
        return G, torch_G, community_label, peak_resolution, nmi_score
        
    def __len__(self):
        return len(self.graphs)


# Input: file_path / testset(file_tree) / device inform
# Output: dataset (obj.)
# Used in: Dataset initialization
class test_graph_loader(torch.utils.data.Dataset):
    def __init__(self, test_path, testset='TC1', device='cpu'):
        self.test_path = test_path
        self.device = device

        if testset == 'real':
            self.test_graphs = glob.glob(os.path.join(test_path, '*-network.dat' ))
            self.test_graphs = natsorted(self.test_graphs)

            self.test_community_labels = glob.glob(os.path.join(test_path, '*-community.dat' ))
            self.test_community_labels = natsorted(self.test_community_labels)    

        else:
            self.test_graphs = glob.glob(os.path.join(test_path, '*-*.dat' ))
            self.test_graphs = [i for i in self.test_graphs if not fnmatch.fnmatch(i, '*-c.dat')]
            self.test_graphs = [i for i in self.test_graphs if i.endswith('dat')]
            self.test_graphs = natsorted(self.test_graphs)

            self.test_community_labels = glob.glob(os.path.join(test_path, '*-*-c.dat' ))
            self.test_community_labels = [i for i in self.test_community_labels if i.endswith('dat')]
            self.test_community_labels = natsorted(self.test_community_labels) 

        self.peak_resolutions = np.loadtxt(os.path.join(test_path,'label.txt'), dtype=str)   
        
    def __getitem__(self, idx):
        test_G = load_network(self.test_graphs[idx])
        test_torch_G = from_networkx(test_G).to(self.device)
        test_torch_G.x = torch.ones((test_torch_G.num_nodes, 1), dtype=torch.float).to(self.device)

        test_community_label = self.test_community_labels[idx]
        _, nmi_score, peak_resolution = self.peak_resolutions[idx]

        nmi_score = torch.tensor([float(nmi_score)]).to(self.device)
        peak_resolution = torch.tensor([float(peak_resolution)]).to(self.device)

        return test_G, test_torch_G, test_community_label, peak_resolution
        
    def __len__(self):
        return len(self.test_graphs)