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

class PairWiseLoss(nn.Module):
	def __init__(self):
		super(PairWiseLoss,self).__init__()
		self.loss = nn.MSELoss()

	def forward(self,features,shortest_path_length,mask):
		n = features.size()[0]
		norm = torch.sum(features,dim=1,keepdim=True)
		norm = norm.expand(n,n)
		distance = norm+norm.t()-2*features.mm(features.t())
		return self.loss(distance*mask,shortest_path_length)

def train(GCN,args,G,F,A,P,M,E):
	device = torch.device("cuda:0" if args.cuda else "cpu")
	optimizer = optim.Adam(GCN.parameters(), lr=args.lr,betas=(0.9,0.999))

	Loss = PairWiseLoss()
	for itera in range(1,args.epochs+1):
		print("==========="+str(itera)+"===========")
		loss = 0
		x = time.time()
		for i in range(0,len(G)):
			g = G[i]
			f = F[i]
			a = A[i]
			p = P[i]
			m = M[i]
			e = E[i]
			
			if args.cuda:
				f = f.cuda()
				a = a.cuda()
				p = p.cuda()
				m = m.cuda()
				e = e.cuda()

			node_features = GCN(g,f)

			gcn_loss = Loss(node_features,p,m)

			loss += gcn_loss.item()
			optimizer.zero_grad()
			gcn_loss.backward()
			optimizer.step()

		y = time.time()
		print("Time = "+str(y-x))
		print("Loss = "+str(loss))

		if itera%100==0:
			torch.save(GCN.state_dict(),args.path+args.dataset+'-'+'epochs-'+str(itera)+'.pth')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 40 epochs"""
    lr = args.lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate(itera,args):
	G = GCN()
	G.load_state_dict(torch.load(path+'/model/'+args.dataset+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-loss-'+args.loss+'-init-'+args.init+'-GCN.pth'))
	t = 0
	file_path = path+'Data/'+args.dataset
	for id in Inference[args.dataset]:
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
		g = dgl.DGLGraph()
		g.add_nodes(num_of_nodes)
		for i in range(0,num_of_nodes):
			for j in range(0,i):
				if adjacency_matrix[i][j]!=0:
					g.add_edge(i,j)
		g.add_edges(g.nodes(), g.nodes())
		features = torch.FloatTensor(features)
		x = time.time()
		with torch.no_grad():
			node_features = G(g,features)
		y = time.time()
		print('Inference Time = '+str(y-x))
		t += (y-x)
		features = node_features.numpy()
		features = np.asarray(features,dtype='<f')
		features = features.flatten('F')
		features.tofile(path+'Result/Result/'+args.dataset+'-node-features-'+'{:03d}'.format(id)+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-loss-'+args.loss+'-init-'+args.init+'.dat',format='<f')

def inference(itera,args):
	if args.model == 'SurfNet':
		G = SurfNet(3)
	elif args.model == 'GMM':
		G = GMM(3)
	elif args.molde == "Edge":
		G = Edge(3)
	G.load_state_dict(torch.load(path+'/model/'+args.dataset+'-'+'epochs-'+str(itera)+'-'+args.model+'.pth'))
	t = 0
	file_path = path+'Data/'+args.dataset
	for id in range(1,2001):
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
		g = dgl.DGLGraph()
		g.add_nodes(num_of_nodes)
		for i in range(0,num_of_nodes):
			for j in range(0,i):
				if adjacency_matrix[i][j]!=0:
					g.add_edge(i,j)
		g.add_edges(g.nodes(), g.nodes())
		features = torch.FloatTensor(features)
		x = time.time()
		with torch.no_grad():
			node_features = G(g,features)
		y = time.time()
		print('Inference Time = '+str(y-x))
		t += (y-x)
		features = node_features.numpy()
		features = np.asarray(features,dtype='<f')
		features = features.flatten('F')
		features.tofile(path+'fuse/'+args.dataset+'-node-features-'+'{:03d}'.format(id)+'-'+'epochs-'+str(itera)+'-samples-'+str(args.samples)+'-loss-'+args.loss+'-init-'+args.init+'.dat',format='<f')