import argparse
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from utils.utils import extract_meta_info

class subnet_MLP(nn.Module):
    def __init__(self, args):
        super(subnet_MLP, self).__init__()
        self.linear1 = nn.Linear(args.sMLP_idim, args.sMLP_hdim1)
        self.linear2 = nn.Linear(args.sMLP_hdim1, args.sMLP_hdim2)
        self.output = nn.Linear(args.sMLP_hdim2, args.sMLP_odim)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.linear2(x)
        x = self.leaky_relu(x)
        x = self.output(x)
        return x 

class subnet_GCN(nn.Module):
    def __init__(self, args):
        super(subnet_GCN, self).__init__()
        self.conv1 = GCNConv(1, args.GCN_hdim1)  
        self.conv2 = GCNConv(args.GCN_hdim1, args.GCN_hdim2)
        self.conv3 = GCNConv(args.GCN_hdim2, args.GCN_odim)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = self.leaky_relu(x)
        x = self.conv3(x, edge_index)
        x = torch.mean(x, dim=0)

        return x
        
class finnet_MLP(nn.Module):
    def __init__(self, args):
        super(finnet_MLP, self).__init__()
        self.linear1 = nn.Linear(args.fMLP_idim, args.fMLP_hdim1)
        self.linear2 = nn.Linear(args.fMLP_hdim1, args.fMLP_hdim2)
        self.output = nn.Linear(args.fMLP_hdim2, args.fMLP_odim)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.linear2(x)
        x = self.leaky_relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x 

class mainnet_ResolNet(nn.Module):
    def __init__(self, args):
        super(mainnet_ResolNet, self).__init__()
        self.args = args
        self.sub_MLP = subnet_MLP(self.args).to(args.device) 
        self.sub_GCN = subnet_GCN(self.args).to(args.device)
        self.fin_MLP = finnet_MLP(self.args).to(args.device)

    def forward(self, G, torch_G):
        meta_info = extract_meta_info(G, self.args.device)
        out1 = self.sub_MLP(meta_info) #10d vector
        out1 = out1.squeeze()

        x = torch_G.x.to(self.args.device)
        edge_index = torch_G.edge_index.to(self.args.device)
        out2 = self.sub_GCN(x, edge_index) #10d vector , batch

        out3 = self.fin_MLP(torch.cat((out1, out2), dim=0))

        return out3


class onlyMLP_ResolNet(nn.Module):
    def __init__(self, args):
        super(onlyMLP_ResolNet, self).__init__()
        self.args = args
        self.args.sMLP_odim = 20

        self.sub_MLP = subnet_MLP(self.args).to(args.device) 
        #self.sub_GCN = subnet_GCN(self.args).to(args.device)
        self.fin_MLP = finnet_MLP(self.args).to(args.device)

    def forward(self, G, torch_G):
        meta_info = extract_meta_info(G, self.args.device)
        out1 = self.sub_MLP(meta_info) #10d vector
        out1 = out1.squeeze()

        out3 = self.fin_MLP(out1)

        return out3


class onlyGCN_ResolNet(nn.Module):
    def __init__(self, args):
        super(onlyGCN_ResolNet, self).__init__()
        self.args = args
        self.args.GCN_odim = 20

        #self.sub_MLP = subnet_MLP(self.args).to(args.device) 
        self.sub_GCN = subnet_GCN(self.args).to(args.device)
        self.fin_MLP = finnet_MLP(self.args).to(args.device)

    def forward(self, G, torch_G):
        #graph_info = torch_G
        x = torch_G.x.to(self.args.device)
        edge_index = torch_G.edge_index.to(self.args.device)
        #batch = torch_G.batch.to(self.args.device)
        out2 = self.sub_GCN(x, edge_index) #10d vector , batch

        out3 = self.fin_MLP(out2)

        return out3