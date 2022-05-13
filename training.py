#!/usr/bin/env python
# coding: utf-8

import numpy as np
import networkx as nx
import nxmetis
import scipy
from scipy.sparse import coo_matrix
from scipy.io import mmread
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, avg_pool, graclus
from torch_geometric.utils import to_networkx, degree, to_scipy_sparse_matrix, get_laplacian, remove_self_loops
from itertools import combinations
import os
import timeit
import random

# Cpu by default
device='cpu'
print('Device:',device)
print('Pytorch version:',torch.__version__)
print('')

# Seeds
torch.manual_seed(176364)
np.random.seed(453658)
random.seed(41884)
       
# Some functions

# Define the graph of a Delaunay mesh. The input is a set of points and the output is the networkx graph associated with the Delaunay
# mesh
def graph_delaunay_from_points(points):
    mesh=Delaunay(points,qhull_options="QJ")
    mesh_simp=mesh.simplices
    edges=[]
    for i in range(len(mesh_simp)):
        edges+=combinations(mesh_simp[i],2)
    e=list(set(edges))
    g=nx.Graph(e) 
    return g

# Creates the dataset made of Delaunay squares and rectangles, FEM triangulations and SuiteSparse graphs
def mixed_dataset(n,n_min,n_max,n_iter,n_iter_suite):
	dataset=[]
	for i in range(int(n/2)):
		num_nodes=np.random.choice(np.arange(n_min,n_max+1,2))
		points=np.random.random_sample((num_nodes,2))
		twos=np.full((num_nodes,1),2)
		ones=np.ones((num_nodes,1))
		resize=np.concatenate((twos,ones),axis=1)
		points=resize*points
		g=graph_delaunay_from_points(points)
		dataset.append(nx.to_scipy_sparse_matrix(g,format='coo',dtype=float))
	for i in range(n):
		num_nodes=np.random.choice(np.arange(n_min,n_max+1,2))
		points=np.random.random_sample((num_nodes,2))
		g=graph_delaunay_from_points(points)
		dataset.append(nx.to_scipy_sparse_matrix(g,format='coo',dtype=float))
	count=0
	count_suite=0
	for m in os.listdir(os.path.expanduser('~/dl-spectral-graph-partitioning/graded_l/')):
		adj=mmread(os.path.expanduser('~/dl-spectral-graph-partitioning/graded_l/'+str(m)))
		if adj.shape[0]<n_max and adj.shape[0]>n_min:
			for i in range(n_iter):
				dataset.append(adj)
				count+=1
	for m in os.listdir(os.path.expanduser('~/dl-spectral-graph-partitioning/hole3/')):
		adj=mmread(os.path.expanduser('~/dl-spectral-graph-partitioning/hole3/'+str(m)))
		if adj.shape[0]<n_max and adj.shape[0]>n_min:
			for i in range(n_iter):
				dataset.append(adj)
				count+=1
	for m in os.listdir(os.path.expanduser('~/dl-spectral-graph-partitioning/hole6/')):
		adj=mmread(os.path.expanduser('~/dl-spectral-graph-partitioning/hole6/'+str(m)))
		if adj.shape[0]<n_max and adj.shape[0]>n_min:
			for i in range(n_iter):
				dataset.append(adj)
				count+=1
	for m in os.listdir(os.path.expanduser('~/dl-spectral-graph-partitioning/suitesparse/')):
		adj=mmread(os.path.expanduser('~/dl-spectral-graph-partitioning/suitesparse/'+str(m)))
		g=nx.Graph(adj)
		if adj.shape[0]<n_max and adj.shape[0]>n_min and nx.is_connected(g):
			for i in range(n_iter_suite):
				dataset.append(adj)
				count_suite+=1
	return dataset,count,count_suite

# Returns the left-normalized laplacian of the input graph
def laplacian(graph):
    lap=get_laplacian(graph.edge_index,num_nodes=graph.num_nodes)
    L=torch.sparse_coo_tensor(lap[0],lap[1]).to(device)
    D=torch.sparse_coo_tensor(torch.stack((torch.arange(graph.num_nodes),torch.arange(graph.num_nodes)),dim=0),lap[1][-graph.num_nodes:]).to(device)
    Dinv=torch.pow(D,-1)
    return torch.sparse.mm(Dinv,L)

# Returns the Rayleigh quotient of the vector x and the matrix L (this will be the left-normalized laplacian in our case)
def rayleigh_quotient(x,L):
    return (torch.t(x).matmul(L.matmul(x))/(torch.t(x).matmul(x)))

# Computes the sum of the eigenvector residual and the eigenvalue related to the vector x
def residual(x,L,mse):
    return mse(L.matmul(x),rayleigh_quotient(x,L)*x)+rayleigh_quotient(x,L)

# Returns the sum of the residual of the computed eigenvectors plus the sum of the eigenvalues. This will be the loss function to train the embedding module)
def loss_embedding(x,L):
    mse=nn.MSELoss()
    l=torch.tensor(0.).to(device)
    for i in range(x.shape[1]):
        l+=residual(x[:,i],L,mse)
    return l.to(device)
	
# Loss function for the partitioning module
def loss_normalized_cut(y_pred, graph):
    y = y_pred
    d = degree(graph.edge_index[0], num_nodes=y.size(0))
    gamma = y.t() @ d
    c = torch.sum(y[graph.edge_index[0], 0] * y[graph.edge_index[1], 1])
    return torch.sum(torch.div(c, gamma)) 
    
##############################################################
#  EMBEDDING MODULE
##############################################################



# Neural network for the embedding module
class ModelSpectral(torch.nn.Module):
    def __init__(self):
        super(ModelSpectral, self).__init__()

        self.l =32
        self.pre = 2
        self.post = 2
        self.coarsening_threshold = 2
        self.activation = torch.tanh
        self.lins=[16,32,32,16,16]
        
        self.conv_post = nn.ModuleList(
            [SAGEConv(self.l, self.l) for i in range(self.post)]
        )
        self.conv_coarse = SAGEConv(2,self.l)

        self.lins1=nn.Linear(self.l,self.lins[0])
        self.lins2=nn.Linear(self.lins[0],self.lins[1]) 
        self.lins3=nn.Linear(self.lins[1],self.lins[2]) 
        self.final=nn.Linear(self.lins[2],2)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        unpool_info = []
        x_info=[]
        cluster_info=[]
        edge_info=[]
        while x.size()[0] > self.coarsening_threshold:
            cluster = graclus(edge_index,num_nodes=x.shape[0])
            cluster_info.append(cluster)
            edge_info.append(edge_index)
            gc = avg_pool(cluster, Batch(batch=batch, x=x, edge_index=edge_index))
            x, edge_index, batch = gc.x, gc.edge_index, gc.batch
        # coarse iterations
        x=torch.eye(2).to(device)
        x=self.conv_coarse(x,edge_index)
        x=self.activation(x)
        while edge_info:
            # un-pooling / interpolation / prolongation / refinement
            edge_index = edge_info.pop()
            output, inverse = torch.unique(cluster_info.pop(), return_inverse=True)
            x = x[inverse]
            # post-smoothing
            for i in range(self.post):
                x = self.activation(self.conv_post[i](x, edge_index))
        x=self.lins1(x)
        x=self.activation(x)
        x=self.lins2(x)
        x=self.activation(x)
        x=self.lins3(x)
        x=self.activation(x)
        x=self.final(x)
        x,_=torch.linalg.qr(x,mode='reduced')
        return x


print('Start embedding module training')
print('')

f=ModelSpectral().to(device)
print('Number of parameters:',sum(p.numel() for p in f.parameters()))
print('')

loss_fn=loss_embedding

# Define the dataset
ng=2000
n_iter,n_iter_suite=15,3

listData,count_fem,count_suite=mixed_dataset(ng,100,5000,n_iter,n_iter_suite)

dataset=[]
for adj in listData:
    row=adj.row
    col=adj.col
    rowcols=np.array([row,col])
    edges=torch.tensor(rowcols,dtype=torch.long)
    nodes=torch.randn(adj.shape[0],2)
    dataset.append(Data(x=nodes, edge_index=edges))
loader=DataLoader(dataset,batch_size=1,shuffle=True)
print('Training dataset for embedding module done')
print('Number of graphs in the training dataset:',len(loader))
print('Number of Delaunay graphs:',ng+int(ng/2))
print('Number of fem graphs:',count_fem)
print('Number of SuiteSparse graphs:',count_suite)
print('')

lr=0.0001 # learning rate
optimizer=torch.optim.Adam(f.parameters(),lr=lr) # optimizer
epochs=120 # epochs                                            
losses=[]
update=torch.tensor(5).to(device) # steps after which the loss function is updated
print_loss=20 # steps after which the loss function is printed #20

# Training loop
print('Start training')
for i in range(epochs):
    loss=torch.tensor(0.).to(device)
    j=0
    for d in loader:
        d=d.to(device)
        L=laplacian(d).to(device)
        x=f(d)
        loss+=loss_fn(x,L)/update
        j+=1
        if j%update.item()==0 or j==len(loader):
            optimizer.zero_grad()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            loss=torch.tensor(0.).to(device)
    if i%print_loss==0:
        print('Epoch:',i,'   Loss:',losses[-1])

print('End training')
print('')

# Save the model
torch.save(f.state_dict(), os.path.expanduser('~/dl-spectral-graph-partitioning/spectral_weights'))
print('Model saved')
print('')

f.eval()
for p in f.parameters():
	p.requires_grad=False
f.eval();

##############################################################
#  PARTITIONING MODULE
##############################################################


print('Start partitioning module training')
print('')

# Neural network for the partitioning module
class ModelPartitioning(torch.nn.Module):
	def __init__(self):
		super(ModelPartitioning, self).__init__()

		self.l =16
		self.pre = 2
		self.post = 2
		self.coarsening_threshold = 2
		self.activation = torch.tanh
		self.lins=[16,16,16,16,16]

		self.conv_first = SAGEConv(1, self.l)
		self.conv_pre = nn.ModuleList(
		    [SAGEConv(self.l, self.l) for i in range(self.pre)]
		)
		self.conv_post = nn.ModuleList(
		    [SAGEConv(self.l, self.l) for i in range(self.post)]
		)
		self.conv_coarse = SAGEConv(self.l,self.l)

		self.lins1=nn.Linear(self.l,self.lins[0])
		self.lins2=nn.Linear(self.lins[0],self.lins[1]) 
		self.lins3=nn.Linear(self.lins[1],self.lins[2]) 
		self.final=nn.Linear(self.lins[4],2)

	def forward(self, graph):
		x, edge_index, batch = graph.x, graph.edge_index, graph.batch
		x = self.activation(self.conv_first(x, edge_index))
		unpool_info = []
		x_info=[]
		cluster_info=[]
		edge_info=[]
		batches=[]
		while x.size()[0] > self.coarsening_threshold:
		    # pre-smoothing
		    for i in range(self.pre):
		        x = self.activation(self.conv_pre[i](x, edge_index))
		    # pooling / coarsening / restriction
		    x_info.append(x)
		    batches.append(batch)
		    cluster = graclus(edge_index,num_nodes=x.shape[0])
		    cluster_info.append(cluster)
		    edge_info.append(edge_index)
		    gc = avg_pool(cluster, Batch(batch=batch, x=x, edge_index=edge_index))
		    x, edge_index, batch = gc.x, gc.edge_index, gc.batch
		# coarse iterations
		x = self.activation(self.conv_coarse(x,edge_index))
		while edge_info:
			# un-pooling / interpolation / prolongation / refinement
			edge_index = edge_info.pop()
			output, inverse = torch.unique(cluster_info.pop(), return_inverse=True)
			x = (x[inverse] + x_info.pop())/2
		    # post-smoothing
			for i in range(self.post):
				x = self.activation(self.conv_post[i](x, edge_index))
		x=self.lins1(x)
		x=self.activation(x)
		x=self.lins2(x)
		x=self.activation(x)
		x=self.lins3(x)
		x=self.activation(x)
		x=self.final(x)
		x=torch.softmax(x,dim=1)
		return x
		
f_lap=ModelPartitioning().to(device) 
print('Number of parameters:',sum(p.numel() for p in f_lap.parameters()))
print('')

# Define the dataset
ng=20
n_iter=5
n_iter_suite=1

listData,count,count_suite=mixed_dataset(ng,200,500,n_iter,n_iter_suite)

dataset=[]
for adj in listData:

	row=adj.row
	col=adj.col
	rowcols=np.array([row,col])
	edges=torch.tensor(rowcols,dtype=torch.long)
	edges,_=remove_self_loops(edges)
	nodes=torch.randn(adj.shape[0],2)

	graph=Batch(x=nodes, edge_index=edges).to(device)
	graph.x=f(graph)[:,1].reshape((graph.num_nodes,1))
	graph.x=(graph.x-torch.mean(graph.x))*torch.sqrt(torch.tensor(graph.num_nodes))
	dataset.append(graph)
	j+=1

print('Number of graphs in the training dataset:',len(listData))
print('Number of FEM graphs:',count)
print('Number of SuiteSparse graphs:',count_suite)
loader=DataLoader(dataset,batch_size=1,shuffle=True,pin_memory=False)
print('Training dataset for partitioning module done')
print('')

loss_fn=loss_normalized_cut 

lr=0.0001 # learning rate
optimizer=torch.optim.Adam(f_lap.parameters(),lr=lr) # optimizer
epochs=500 # epochs                                                   
losses=[]
update=torch.tensor(5).to(device) # steps after which the loss function is updated
print_loss=50 # steps after which the loss function is printed

# Training loop
print('Start training')
for i in range(epochs):
    loss=torch.tensor(0.).to(device)
    j=0
    for d in loader:
        d=d.to(device)
        data=f_lap(d)
        loss+=loss_fn(data,d)/update
        j+=1
        if j%update.item()==0 or j==len(loader):
            optimizer.zero_grad()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            loss=torch.tensor(0.).to(device)
    if i%print_loss==0:
        print('Epoch:',i,'   Loss:',losses[-1])
        
print('End training')
print('')

torch.save(f_lap.state_dict(), os.path.expanduser('~/dl-spectral-graph-partitioning/partitioning_weights'))
print('Model saved')
