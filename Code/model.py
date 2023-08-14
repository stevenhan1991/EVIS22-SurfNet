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
from dgl.nn.pytorch import GraphConv,GMMConv, EdgeConv
from dgl import DGLGraph
import networkx as nx


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find("Conv")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("Linear")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("BatchNorm")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

class SurfNet(nn.Module):
    def __init__(self,input_dim=3):
        super(SurfNet, self).__init__()
        self.ec1 = GraphConv(input_dim,64,activation=F.relu)
        self.ec2 = GraphConv(64,128,activation=F.relu)
        self.ec3 = GraphConv(128,256,activation=F.relu)
        self.ec4 = GraphConv(256,256,activation=F.relu)
        self.ec5 = GraphConv(256,256,activation=F.relu)
        self.ec6 = GraphConv(256,256,activation=F.relu)

    def forward(self, g, features):
    	x1 = self.ec1(g,features)
    	x2 = self.ec2(g,x1)
    	x3 = self.ec3(g,x2)
    	x4 = self.ec4(g,x3)
    	x5 = self.ec5(g,x4+x3)
    	x6 = self.ec6(g,x3+x4+x5)
    	return x6


class GMM(nn.Module):
	def __init__(self,input_dim=3):
		super(GMM,self).__init__()
		self.conv1 = GMMConv(input_dim,64,3,2,'mean')
		self.conv2 = GMMConv(64,128,3,2,'mean')
		self.conv3 = GMMConv(128,256,3,2,'mean')
		self.conv4 = GMMConv(256,256,3,2,'mean')

	def forward(self,g,features,edge_attr):
		x1 = self.conv1(g,features,edge_attr)
		x2 = self.conv2(g,x1,edge_attr)
		x3 = self.conv3(g,x2,edge_attr)
		x4 = self.conv4(g,x3,edge_attr)
		return x4


