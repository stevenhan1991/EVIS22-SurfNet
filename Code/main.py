import numpy as np
import os
import time
import argparse
import copy
import sys
from train import *
from model import *
from DataPre import *

parser = argparse.ArgumentParser(description='PyTorch Implementation of V2V')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate of G')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training')
parser.add_argument('--epochs', type=int,default=1000,
                    help='number of epochs to train (default: 500)')
parser.add_argument('--samples', type=int, default=1000,
                    help='number of training samples')
parser.add_argument('--dataset', type=str,
                    help='')
parser.add_argument('--init', type=str, default='pos',
                    help='')
parser.add_argument('--mode', type=str, default = 'train',
                    help='')
parser.add_argument('--path', type=str, default = '../Data/',
                    help='The path where we save data')

args = parser.parse_args()
print(not args.no_cuda)
print(torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 30, 'pin_memory': True} if args.cuda else {}



def GenerateAdjList(adjacency_matrix):
	adj_list = []
	for i in range(0,len(adjacency_matrix)):
		nei_nodes = []
		for j in range(0,len(adjacency_matrix)):
			if j!=i and adjacency_matrix[i][j]==1:
				nei_nodes.append(j)
		adj_list.append(nei_nodes)
	return adj_list


def GetKOrderAdjacencyMatrix(adjacency_matrix,k):
	a = np.copy(adjacency_matrix)
	for i in range(1,k):
		a = np.matmul(a,adjacency_matrix.transpose())
	return a 

def main():
	G = []
	F = []
	A = []
	P = []
	M = []
	E = []
	file_path = args.path+args.dataset
	if os.path.exists(file_path+'/adjacency-matrix-'+'{:03d}'.format(iso)+'.graphml'):
		g, features, adjacency_matrix, mask, paths, edge_attr = InitGraphStream(args,file_path)
		G.append(g)
		F.append(features)
		A.append(adjacency_matrix)
		P.append(paths)
		M.append(mask)
		E.append(edge_attr)
	model = SurfNet(3)
	if args.cuda:
		model.cuda()
	train(model,args,G,F,A,P,M,E)

if __name__== "__main__":
	if args.mode =='train':
		main()
	else:
		inference(args.epochs)