import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import argparse
import copy
import torch.optim as optim
import sys
from torch.nn import init
import dgl
from dgl.nn.pytorch import GraphConv
from dgl import DGLGraph
import networkx as nx

def InitGraphStream(args,file_path,id):
	graph = nx.read_graphml(file_path+'/adjacency-matrix-'+'{:03d}'.format(id)+'.graphml')
	num_of_nodes = nx.number_of_nodes(graph)
	adjacency_matrix = nx.to_numpy_matrix(graph)
	adjacency_matrix = np.asarray(adjacency_matrix,dtype='<f')
	if args.init == 'pos' or args.init == 'vec' or args.init == 'pos+vec':
		features = np.fromfile(file_path+'/nodes-features-'+'{:03d}'.format(id)+'.dat',dtype='<f')
	elif args.init == 'normal':
		features = np.fromfile(file_path+'/nodes-features-normals-'+'{:03d}'.format(id)+'.dat',dtype='<f')
	features = features.reshape(6,num_of_nodes).transpose()
	if args.init == 'pos':
		features = features[:,0:3]
	elif args.init == 'vec' or args.init == 'normal':
		features = features[:,3:6]
	elif args.init == 'pos+vec':
		features = features[:,0:6]
	shortest_path_length =  np.fromfile(file_path+'/shortest-path-'+'{:03d}'.format(id)+'.dat',dtype='int16')
	G = dgl.DGLGraph()
	G.add_nodes(num_of_nodes)
	for i in range(0,num_of_nodes):
		for j in range(0,i):
			if adjacency_matrix[i][j]!=0:
				G.add_edge(i,j)
				G.add_edge(j,i)
	G.add_edges(G.nodes(), G.nodes())
	max_coor = np.max(features,axis=0)
	min_coor = np.min(features,axis=0)
	edges = G.edges()
	num_of_edges = len(edges[0])
	edges_attr = np.zeros((num_of_edges,3))
	for i in range(0,num_of_edges):
		n1 = int(edges[0][i])
		n2 = int(edges[1][i])
		edges_attr[i] = 0.5+(features[n2]-features[n1])/(2*(max_coor-min_coor))

	mask = np.zeros((num_of_nodes,num_of_nodes))
	paths = np.zeros((num_of_nodes,num_of_nodes))
	idx = 0
	for k in range(num_of_nodes):
		for l in range(k):
			mask[k][l] = 1
			paths[k][l] = shortest_path_length[idx]
			idx += 1
	return G,torch.FloatTensor(features),torch.FloatTensor(adjacency_matrix),torch.FloatTensor(mask),torch.FloatTensor(paths), torch.FloatTensor(edges_attr)

def InitGraphIso(args,file_path,iso):
	graph = nx.read_graphml(file_path+'/adjacency-matrix-'+'{:03d}'.format(iso)+'.graphml')
	num_of_nodes = nx.number_of_nodes(graph)
	adjacency_matrix = nx.to_numpy_matrix(graph)
	adjacency_matrix = np.asarray(adjacency_matrix,dtype='<f')
	features = np.fromfile(file_path+'/nodes-features-'+'{:03d}'.format(iso)+'.dat',dtype='<f')
	features = features.reshape(3,num_of_nodes).transpose()
	shortest_path_length =  np.fromfile(file_path+'/shortest-path-'+'{:03d}'.format(iso)+'.dat',dtype='int16')
	in_nodes = []
	out_nodes = []
	for i in range(0,num_of_nodes):
		for j in range(0,i):
			if adjacency_matrix[i][j]!=0:
				in_nodes.append(i)
				out_nodes.append(j)
	G = dgl.graph((torch.tensor(in_nodes),torch.tensor(out_nodes)))
	G = dgl.to_bidirected(G)
	G = dgl.add_self_loop(G)
	max_coor = np.max(features,axis=0)
	min_coor = np.min(features,axis=0)
	edges = G.edges()
	num_of_edges = len(edges[0])
	edges_attr = np.zeros((num_of_edges,3))
	for i in range(0,num_of_edges):
		n1 = int(edges[0][i])
		n2 = int(edges[1][i])
		edges_attr[i] = 0.5+(features[n2]-features[n1])/(2*(max_coor-min_coor))

	mask = np.zeros((num_of_nodes,num_of_nodes))
	paths = np.zeros((num_of_nodes,num_of_nodes))
	idx = 0
	for k in range(num_of_nodes):
		for l in range(k):
			mask[k][l] = 1
			paths[k][l] = shortest_path_length[idx]
			idx += 1
	return G,torch.FloatTensor(features),torch.FloatTensor(adjacency_matrix),torch.FloatTensor(mask),torch.FloatTensor(paths), torch.FloatTensor(edges_attr)
