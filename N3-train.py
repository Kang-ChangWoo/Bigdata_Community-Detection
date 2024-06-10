
import os, argparse, random, glob

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#from precompute_peak import *

from utils.utils import load_network, load_ground_truth
from utils.utils import extract_meta_info, calculate_nmi
from utils.dataloader import train_graph_loader, test_graph_loader
from utils.detection import detect_communities_louvain, detect_communities_leiden
from utils.network import mainnet_ResolNet

def validate(net, test_loader):
    criterion = nn.MSELoss()
    data_iter = len(test_loader)

    test_nmi_score = 0.0

    with torch.no_grad():
        for i, (test_G, test_torch_G, test_community_label, peak_resolution) in enumerate(test_loader, 0):

            out = net(test_G, test_torch_G)

            # Scaling output
            min_value = nn.Parameter(torch.tensor([0.001])).to(args.device)
            range_value = nn.Parameter(torch.tensor([50.0])).to(args.device)
            out = min_value + range_value * out

            eval_loss = criterion(out, peak_resolution).item() / data_iter # + nmi_loss
            
            """
            # For training efficiency, we deprecate nmi loss

            out_new = out.item()
            predicted_community = detect_communities_louvain(test_G, out_new)

            true_communities = load_ground_truth(test_community_label)
            test_nmi_score += calculate_nmi(true_communities, predicted_community)
            """

            print("Loss", eval_loss)

        print(f'Test done. (Test nmi score: {eval_loss:.8f})')

    return eval_loss



def train(args):
    # seed fixed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_loader = train_graph_loader(args.train_path, args.device)
    test_loader = test_graph_loader(args.test_path, args.testset, args.device)
    
    model = mainnet_ResolNet(args).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
    criterion = nn.MSELoss()

    print("lets go!")
    best_performance = - np.inf

    for epoch in range(args.epoch):
        model.train()
        # r = 1.0

        # TODO= add up tqdm
        for i, (G, torch_G, community_label, peak_resolution, nmi_score) in tqdm(enumerate(train_loader, 0), desc=f'{epoch}-th learning', total=len(train_loader)):
            optimizer.zero_grad()

            out = model(G, torch_G)

            # Scaling output
            min_value = nn.Parameter(torch.tensor([000.1])).to(args.device)
            range_value = nn.Parameter(torch.tensor([50.0])).to(args.device)
            out = min_value + range_value * out 
            print(out)
    
            """
            # For training efficiency, we deprecate nmi loss

            # out = out.item()
            # predicted_community = detect_communities_louvain(G, resolution_=out)

            # gt_communities = load_ground_truth(community_label)
            # pred_nmi_score = calculate_nmi(gt_communities, predicted_community)
            # pred_nmi_score = torch.tensor([float(pred_nmi_score)]).to(args.device)
            # nmi_loss = criterion(nmi_score, pred_nmi_score)
            """

            res_loss = criterion(out, peak_resolution)
            loss = res_loss # + nmi_loss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            print(f'Epoch [{epoch+1}/{args.epoch}], Loss: {loss.item():.4f}')

        model.eval()
        eval_performance = validate(model, test_loader)
        
        if eval_performance > best_performance:
            best_performance = eval_performance
            torch.save(model, f'./ckpt/{args.pth}-best.pt')

    torch.save(model, f'./ckpt/{args.pth}-last.pt')

if __name__ == "__main__":
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Related to path
    parser.add_argument('--train_path', type=str, default='path/to/yours', help='path of training dataset')
    parser.add_argument('--test_path', type=str, default='path/to/yours', help='path of test dataset')
    parser.add_argument('--testset', type=str, default='real', help='-')
    
    # Training parameters
    parser.add_argument('--epoch', type=int, default=150, help='epoch')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--pth', type=str, default='correction', help='-')

    # Dimension for first MLP (sub)
    parser.add_argument('--sMLP_idim', type=int, default=7, help='dimension of input tensor') 
    parser.add_argument('--sMLP_hdim1', type=int, default=256, help='dimension of 1st MLP hidden layer') 
    parser.add_argument('--sMLP_hdim2', type=int, default=32, help='dimension of 2nd MLP hidden layer') 
    parser.add_argument('--sMLP_odim', type=int, default=10, help='dimension of output tensor') 

    # Dimension for GCN 
    # input dimension fixed as 1
    parser.add_argument('--GCN_hdim1', type=int, default=64, help='dimension of 1st GCN hidden layer') 
    parser.add_argument('--GCN_hdim2', type=int, default=64, help='dimension of 2nd GCN hidden layer') 
    parser.add_argument('--GCN_odim', type=int, default=10, help='dimension of output tensor') 

    # Dimension for final MLP
    parser.add_argument('--fMLP_idim', type=int, default=20, help='dimension of input tensor') 
    parser.add_argument('--fMLP_hdim1', type=int, default=256, help='dimension of 1st MLP hidden layer') 
    parser.add_argument('--fMLP_hdim2', type=int, default=32, help='dimension of 2nd MLP hidden layer') 
    parser.add_argument('--fMLP_odim', type=int, default=1, help='dimension of output tensor') 

    args = parser.parse_args()
    train(args)