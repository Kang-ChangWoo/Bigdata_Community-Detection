import argparse
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from utils.utils import extract_meta_info

def get_args():
    parser = argparse.ArgumentParser(description="프로그램 설명")
    parser.add_argument('--model_path', type=str, required=True, help='모델 경로')
    parser.add_argument('--ab_type', type=str, required=True, help='Ablation 스터디 타입')  # ab_type 추가
    parser.add_argument('--sMLP_idim', type=int, default=100)
    parser.add_argument('--sMLP_hdim1', type=int, default=50)
    parser.add_argument('--sMLP_hdim2', type=int, default=25)
    parser.add_argument('--sMLP_odim', type=int, default=10)
    parser.add_argument('--GCN_hdim1', type=int, default=64)
    parser.add_argument('--GCN_hdim2', type=int, default=32)
    parser.add_argument('--GCN_odim', type=int, default=10)
    parser.add_argument('--fMLP_idim', type=int, default=20)
    parser.add_argument('--fMLP_hdim1', type=int, default=10)
    parser.add_argument('--fMLP_hdim2', type=int, default=5)
    parser.add_argument('--fMLP_odim', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()

args = get_args()
print(f"ab_type: {args.ab_type}")  # args 객체 확인

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
        print(f"ab_type: {self.ablation}")  # ab_type 확인
        self.sub_MLP = subnet_MLP(self.args).to(args.device) 
        self.sub_GCN = subnet_GCN(self.args).to(args.device)
        self.fin_MLP = finnet_MLP(self.args).to(args.device)

    def forward(self, G, torch_G):
        if self.ablation == 'only_mlp':
            meta_info = extract_meta_info(G, self.args.device)
            out1 = self.sub_MLP(meta_info) #10d vector
            out1 = out1.squeeze()
            out3 = self.fin_MLP(out1)

        elif self.ablation == 'only_gcn':
            x = torch_G.x.to(self.args.device)
            edge_index = torch_G.edge_index.to(self.args.device)
            out2 = self.sub_GCN(x, edge_index) #10d vector
            out3 = self.fin_MLP(out2)

        else:
            meta_info = extract_meta_info(G, self.args.device)
            out1 = self.sub_MLP(meta_info) #10d vector
            out1 = out1.squeeze()
            x = torch_G.x.to(self.args.device)
            edge_index = torch_G.edge_index.to(self.args.device)
            out2 = self.sub_GCN(x, edge_index) #10d vector
            out3 = self.fin_MLP(torch.cat((out1, out2), dim=0))

        return out3

model = mainnet_ResolNet(args)