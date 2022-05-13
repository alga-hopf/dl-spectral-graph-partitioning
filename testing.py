#!/usr/bin/env python
# coding: utf-8

import numpy as np
import networkx as nx
import nxmetis
import scipy
from scipy.sparse import coo_matrix
from scipy.io import mmread,mminfo
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, avg_pool, graclus
from torch_geometric.utils import to_networkx, degree, to_scipy_sparse_matrix, get_laplacian, remove_self_loops, subgraph
from itertools import combinations
import os
import timeit
import random
import ctypes
import argparse
from pathlib import Path

libscotch = ctypes.cdll.LoadLibrary('scotch/build/libSCOTCHWrapper.so')

# Some function to generate the dataset. The dataset is made of graphs
# associated with Delaunay meshes

# Define the graph of a Delaunay mesh. The input is a set of points and the output is the networkx graph associated with the Delaunay
# mesh


def graph_delaunay_from_points(points):
    mesh = Delaunay(points, qhull_options="QJ")
    mesh_simp = mesh.simplices
    edges = []
    for i in range(len(mesh_simp)):
        edges += combinations(mesh_simp[i], 2)
    e = list(set(edges))
    g = nx.Graph(e)
    return g

# Returns the pytorch geometric graph partitioned as specified in preds


def torch_from_preds(graph, preds):
    graph_torch = graph
    graph_torch.x = preds
    return graph_torch

# Volumes of the partitioned graph


def volumes(graph):
    ia = torch.where(graph.x == torch.tensor(0.))[0]
    ib = torch.where(graph.x == torch.tensor(1.))[0]
    degs = degree(
    graph.edge_index[0],
    num_nodes=graph.x.size(0),
     dtype=torch.uint8)
    da = torch.sum(degs[ia]).detach().item()
    db = torch.sum(degs[ib]).detach().item()
    cut = torch.sum(graph.x[graph.edge_index[0]] !=
                    graph.x[graph.edge_index[1]]).detach().item() / 2
    return cut, da, db

# Cut of the partitioned graph


def cut(graph):
    cut = torch.sum((graph.x[graph.edge_index[0],
    :2] != graph.x[graph.edge_index[1],
     :2]).all(axis=-1)).detach().item() / 2
    return cut

# Normalized cut of the partitioned graph


def normalized_cut(graph):
    c, dA, dB = volumes(graph)
    if dA == 0 or dB == 0:
        return 2, 0, dA if dA != 0 else dB, c
    else:
        return c / dA + c / dB, dA, dB, c

# Spectral partitioning of the graph networkx graph g relative to the vector x


def nc_eig_median(x, g):
    ind_sort = np.argsort(x)
    ncuts = []
    vols = []
    cuts = []
    void_part = np.array([1.] * ind_sort.shape[0])
    cut, vola, volb = 0., 0., nx.volume(g, g.nodes())
    for i in range(0, g.number_of_nodes() - 1):
        void_part[ind_sort[i]] = 0.
        cut_in, cut_out = 0, 0
        for w in g[ind_sort[i]]:
        	if void_part[w] == 0.:
        		cut_in += 1
        	else:
        		cut_out += 1
        cut = cut - cut_in + cut_out
        deg = g.degree(ind_sort[i])
        vola = vola + deg
        volb = volb - deg
        ncuts.append(cut * (1 / vola + 1 / volb))
        vols.append((vola, volb))
        cuts.append(cut)
    min_nc = np.argmin(np.array(ncuts))
    void_part = np.array([1.] * ind_sort.shape[0])
    void_part[ind_sort[:min_nc]] = torch.tensor(0.)
    return void_part, ncuts[min_nc], vols[min_nc], cuts[min_nc]

# Returns the normalized cut, volumes, partitioning, cut and runtimes of
# the GAP partitioning computed with the embedding module+partitioning
# module (h). We compute it n_iter times and we keep the vector with the
# lowest normalized cut


def best_part(h, graph, n_times):
	ncuts = []
	vols = []
	preds = []
	cuts = []
	t0 = timeit.default_timer()
	graph_ev = h(graph)
	t1 = timeit.default_timer() - t0
	predictions = torch.argmax(graph_ev, dim=1)
	graph_pred = torch_from_preds(graph, predictions)
	nc_gap, vola, volb, cut = normalized_cut(graph_pred)
	ncuts.append(nc_gap)
	cuts.append(cut)
	vols.append((vola, volb))
	preds.append(predictions)
	for i in range(1, n_times):
		t0_loop = timeit.default_timer()
		graph_ev = h(graph)
		t1_loop = timeit.default_timer() - t0_loop
		predictions = torch.argmax(graph_ev, dim=1)
		graph_pred = torch_from_preds(graph, predictions)
		nc_gap, vola, volb, cut = normalized_cut(graph_pred)
		ncuts.append(nc_gap)
		vols.append((vola, volb))
		preds.append(predictions)
		cuts.append(cut)
	min_nc = np.argmin(ncuts)
	return ncuts[min_nc], vols[min_nc], preds[min_nc], cuts[min_nc], t1 + t1_loop

# Returns the normalized cut, volumes, partitioning, cut and runtimes of
# the Approximated spectral partitioning computed with the embedding
# module (f). We compute it n_iter times and we keep the vector with the
# lowest normalized cut


def best_part_eig(f, g, graph, n_iter):
	ncuts = []
	vols = []
	preds = []
	parts = []
	cuts = []
	t0 = timeit.default_timer()
	x = f(graph)[:, 1].reshape(graph.num_nodes).cpu().numpy()
	t1 = timeit.default_timer() - t0
	predictions, nc, volumes, cut = nc_eig_median(x, g)
	ncuts.append(nc)
	vols.append(volumes)
	parts.append(x)
	preds.append(predictions)
	cuts.append(cut)
	for i in range(1, n_iter):
		t01 = timeit.default_timer()
		x = f(graph)[:, 1].reshape(graph.num_nodes).cpu().numpy()
		t11 = timeit.default_timer() - t01
		predictions, nc, volumes, cut = nc_eig_median(x, g)
		ncuts.append(nc)
		vols.append(volumes)
		parts.append(x)
		preds.append(predictions)
		cuts.append(cut)
	min_nc = np.argmin(ncuts)
	return ncuts[min_nc], vols[min_nc], preds[min_nc], parts[min_nc], cuts[min_nc], t1 + t11

# Returns the normalized cut, runtime, volumes and cut of the METIS
# partitioning of the networkx graph g


def normalized_cut_metis(g):
    t0 = timeit.default_timer()
    cut, parts = nxmetis.partition(g, 2)
    t1 = timeit.default_timer() - t0
    degA = sum(g.degree(i) for i in parts[0])
    degB = sum(g.degree(i) for i in parts[1])
    return cut * (1 / degA + 1 / degB), t1, degA, degB, cut

# Returns the normalized cut, runtime, volumes and cut of the Scotch
# partitioning of the networkx graph g


def scotch_partition(g):
    gnx = to_networkx(g, to_undirected=True)
    a = nx.to_scipy_sparse_matrix(gnx, format="csr", dtype=np.float32)
    n = g.num_nodes
    part = np.zeros(n, dtype=np.int32)
    t0 = timeit.default_timer()
    libscotch.WRAPPER_SCOTCH_graphPart(
        ctypes.c_int(n),
        ctypes.c_void_p(a.indptr.ctypes.data),
        ctypes.c_void_p(a.indices.ctypes.data),
        ctypes.c_void_p(part.ctypes.data)
    )
    t1 = timeit.default_timer() - t0
    g.x[np.where(part == 0)] = torch.tensor(0).to(device)
    g.x[np.where(part == 1)] = torch.tensor(1).to(device)
    gr = g.clone().to(device)
    nc, vola, volb, cut = normalized_cut(gr)
    return nc, t1, vola, volb, cut

# Returns the left-normalized laplacian of the input graph


def laplacian(graph):
    lap = get_laplacian(graph.edge_index)
    L = torch.sparse_coo_tensor(lap[0], lap[1]).to(device)
    D = torch.sparse_coo_tensor(torch.stack((torch.arange(graph.num_nodes).to(
        device), torch.arange(graph.num_nodes).to(device)), dim=0), lap[1][-graph.num_nodes:])
    Dinv = torch.pow(D, -1)
    return torch.sparse.mm(Dinv, L)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument(
		"--nmin",
		default=100,
		help="Minimum graph size",
		type=int)
	parser.add_argument(
		"--nmax",
		default=10000,
		help="Maximum graph size",
		type=int)
	parser.add_argument(
		"--ntest",
		default=50,
		help="Number of test graphs",
		type=int)
	parser.add_argument(
		"--dataset",
		default='delaunay',
		help="Dataset type: delaunay, suitesparse, graded_l, hole3, hole6",
		type=str)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('Device:', device)
	print('Pytorch version:', torch.__version__)
	print('')
	
	args = parser.parse_args()

	# Here we set the seeds
	torch.manual_seed(176364)
	np.random.seed(453658)
	random.seed(41884)
	torch.cuda.manual_seed(9597121)

	nmin = args.nmin
	nmax = args.nmax
	ntest = args.ntest
	dataset_type = args.dataset

	# Embedding module
	class ModelSpectral(torch.nn.Module):
		def __init__(self):
			super(ModelSpectral, self).__init__()

			self.l = 32
			self.pre = 2
			self.post = 2
			self.coarsening_threshold = 2
			self.activation = torch.tanh
			self.lins = [16, 32, 32, 16, 16]

			self.conv_post = nn.ModuleList(
			    [SAGEConv(self.l, self.l) for i in range(self.post)]
			)
			self.conv_coarse = SAGEConv(2, self.l)

			self.lins1 = nn.Linear(self.l, self.lins[0])
			self.lins2 = nn.Linear(self.lins[0], self.lins[1])
			self.lins3 = nn.Linear(self.lins[1], self.lins[2])
			self.final = nn.Linear(self.lins[2], 2)

		def forward(self, graph):
			x, edge_index, batch = graph.x, graph.edge_index, graph.batch
			unpool_info = []
			x_info = []
			cluster_info = []
			edge_info = []
			while x.size()[0] > self.coarsening_threshold:
			    cluster = graclus(edge_index, num_nodes=x.shape[0])
			    cluster_info.append(cluster)
			    edge_info.append(edge_index)
			    gc = avg_pool(cluster, Batch(batch=batch, x=x, edge_index=edge_index))
			    x, edge_index, batch = gc.x, gc.edge_index, gc.batch
			# coarse iterations
			x = torch.eye(2).to(device)
			x = self.conv_coarse(x, edge_index)
			x = self.activation(x)
			while edge_info:
			    # un-pooling / interpolation / prolongation / refinement
			    edge_index = edge_info.pop()
			    output, inverse = torch.unique(cluster_info.pop(), return_inverse=True)
			    x = x[inverse]
			    # post-smoothing
			    for i in range(self.post):
			        x = self.activation(self.conv_post[i](x, edge_index))
			x = self.lins1(x)
			x = self.activation(x)
			x = self.lins2(x)
			x = self.activation(x)
			x = self.lins3(x)
			x = self.activation(x)
			x = self.final(x)
			x, _ = torch.linalg.qr(x, mode='reduced')
			return x

	f = ModelSpectral().to(device)
	print('Number of embedding module parameters:', sum(p.numel()
	      for p in f.parameters()))
	print('')
	f.load_state_dict(torch.load(os.path.expanduser(
	    '~/dl-spectral-graph-partitioning/spectral_weights')))

	f.eval()
	for p in f.parameters():
		p.requires_grad = False
	f.eval();

	# Partitioning module
	class ModelPartitioning(torch.nn.Module):
		def __init__(self):
			super(ModelPartitioning, self).__init__()

			self.l = 16
			self.pre = 2
			self.post = 2
			self.coarsening_threshold = 2
			self.activation = torch.tanh
			self.lins = [16, 16, 16, 16, 16]

			self.conv_first = SAGEConv(1, self.l)
			self.conv_pre = nn.ModuleList(
				[SAGEConv(self.l, self.l) for i in range(self.pre)]
			)
			self.conv_post = nn.ModuleList(
				[SAGEConv(self.l, self.l) for i in range(self.post)]
			)
			self.conv_coarse = SAGEConv(self.l, self.l)

			self.lins1 = nn.Linear(self.l, self.lins[0])
			self.lins2 = nn.Linear(self.lins[0], self.lins[1])
			self.lins3 = nn.Linear(self.lins[1], self.lins[2])
			self.final = nn.Linear(self.lins[4], 2)

		def forward(self, graph):
			x, edge_index, batch = graph.x, graph.edge_index, graph.batch
			x = self.activation(self.conv_first(x, edge_index))
			unpool_info = []
			x_info = []
			cluster_info = []
			edge_info = []
			batches = []
			while x.size()[0] > self.coarsening_threshold:
				# pre-smoothing
				for i in range(self.pre):
					x = self.activation(self.conv_pre[i](x, edge_index))
				# pooling / coarsening / restriction
				x_info.append(x)
				batches.append(batch)
				cluster = graclus(edge_index, num_nodes=x.shape[0])
				cluster_info.append(cluster)
				edge_info.append(edge_index)
				gc = avg_pool(cluster, Batch(batch=batch, x=x, edge_index=edge_index))
				x, edge_index, batch = gc.x, gc.edge_index, gc.batch
			# coarse iterations
			x = self.activation(self.conv_coarse(x, edge_index))
			while edge_info:
				# un-pooling / interpolation / prolongation / refinement
				edge_index = edge_info.pop()
				output, inverse = torch.unique(cluster_info.pop(), return_inverse=True)
				x = (x[inverse] + x_info.pop()) / 2
				# post-smoothing
				for i in range(self.post):
					x = self.activation(self.conv_post[i](x, edge_index))
			x = self.lins1(x)
			x = self.activation(x)
			x = self.lins2(x)
			x = self.activation(x)
			x = self.lins3(x)
			x = self.activation(x)
			x = self.final(x)
			x = torch.softmax(x, dim=1)
			return x

	f_lap = ModelPartitioning().to(device)
	print('Number of partitioning module parameters:', sum(p.numel()
	      for p in f_lap.parameters()))
	print('')
	f_lap.load_state_dict(torch.load(os.path.expanduser('~/dl-spectral-graph-partitioning/partitioning_weights')))

	f_lap.eval()
	for p in f_lap.parameters():
		p.requires_grad = False
	f_lap.eval();

	# GAP model (embedding+standardization+partitioning)
	class GAP(torch.nn.Module):
		def __init__(self, eigenvector, part):
			super(GAP, self).__init__()

			self.eig_nn = eigenvector
			self.part_nn = part

		def forward(self, graph):
			eigs = self.eig_nn(graph)[:, 1].reshape(graph.num_nodes, 1)
			graph.x = eigs
			graph.x = (graph.x - torch.mean(graph.x)) * \
			           torch.sqrt(torch.tensor(graph.num_nodes))  # standardization
			return self.part_nn(graph)

	h = GAP(f, f_lap).to(device)

	print('Testing on ' + str(dataset_type) + ' dataset')
	print('')
	list_picked = []
	n_iter = 2
	i = 0
	count_failed = 0
	all_nc, all_nc_metis = [], []
	nodes_graphs, edges_graphs = [], []
	vols, vols_eig, vols_met, vols_eig_true, vols_scotch = [], [], [], [], []
	nc_gap_single, nc_metis_single, nc_scotch_single, nc_eig_single, nc_true_single = [], [], [], [], []
	c_gap_single, c_metis_single, c_scotch_single, c_eig_single, c_true_single = [], [], [], [], []
	t_gap_single, t_metis_single, t_scotch_single, t_eig_single, t_true_single = [], [], [], [], []
	names_failed=[]
	while i < ntest:
		# Creates a graph for testing according to the chosen dataset
		if dataset_type == 'delaunay':

			num_nodes = np.random.choice(np.arange(nmin, nmax + 1))
			points = np.random.random_sample((num_nodes, 2))
			print('Graph:', i)
			if random.random() < 0.5:
				print('Delaunay rectangle')
				twos = np.full((num_nodes, 1), 5)
				ones = np.ones((num_nodes, 1))
				resize = np.concatenate((twos, ones), axis=1)
				points = resize * points
			else:
				print('Delaunay square')
			g=graph_delaunay_from_points(points)
			adj = nx.to_scipy_sparse_matrix(g, format='coo', dtype=float)
			row = adj.row
			col = adj.col
			rowcols = np.array([row,col])
			edges = torch.tensor(rowcols, dtype=torch.long)
			nodes = torch.randn(adj.shape[0], 2)
			g = nx.Graph(adj)
			graph = Batch(x=nodes, edge_index=edges).to(device)
			degrees = list(dict(g.degree()).values())
			print('Number of nodes:', graph.num_nodes)
			print('Number of edges:', graph.num_edges)
			print('Max degree:',max(degrees),'   Min degree:',min(degrees),'   Avg degree:',np.round(np.mean(degrees),4))
			i += 1

		else:
			if len(list_picked) >= len(os.listdir(os.path.expanduser('~/dl-spectral-graph-partitioning/' +str(dataset_type) +'/'))):
			    break
			matrix = random.choice(os.listdir(os.path.expanduser('~/dl-spectral-graph-partitioning/' +str(dataset_type) +'/')))
			if str(matrix) not in list_picked:
				list_picked.append(str(matrix))
				adj = mmread(os.path.expanduser('~/dl-spectral-graph-partitioning/' +str(dataset_type) +'/' +str(matrix)))
				if adj.shape[0]>nmin and adj.shape[0]<nmax:
					_,_,_,_,_,symm=mminfo(os.path.expanduser('~/dl-spectral-graph-partitioning/' +str(dataset_type) +'/'+str(matrix)))
					if symm!='symmetric':
						adj=coo_matrix((1/2)*(adj+adj.transpose()))
					g=nx.Graph(adj)
					
					if nx.is_connected(g):
						row=adj.row
						col=adj.col
						rowcols=np.array([row,col])
						edges,_=remove_self_loops(torch.tensor(rowcols,dtype=torch.long))
						nodes=torch.randn(adj.shape[0],2)
						graph=Batch(x=nodes, edge_index=edges).to(device)
						g=to_networkx(graph,to_undirected=True)
						graph.batch=torch.zeros(graph.num_nodes,dtype=torch.long).to(device)
						degrees=list(dict(g.degree()).values())
						print('Graph: '+str(matrix))
						print('The graph is connected')
						print('Number of nodes:',graph.num_nodes)
						print('Number of edges:',graph.num_edges)
						print('Max degree:',max(degrees),'   Min degree:',min(degrees),'   Avg degree:',np.round(np.mean(degrees),4))
						i+=1
						
					else:
						largest_cc=max(nx.connected_components(g), key=len)
						sub_g_edges,_=subgraph(list(largest_cc),graph.edge_index,relabel_nodes=True)
						edges,_=remove_self_loops(sub_g_edges)
						nodes=torch.randn(len(largest_cc),2)
						graph=Batch(x=nodes, edge_index=edges).to(device)
						g=to_networkx(graph,to_undirected=True)
						degrees=list(dict(g.degree()).values())
						print('Graph: '+str(matrix))
						print('The graph is not connected')
						print('Number of nodes in the largest connected component:',graph.num_nodes)
						print('Number of edges in the largest connected component:',graph.num_edges)
						print('Max degree:',max(degrees),'   Min degree:',min(degrees),'   Avg degree:',np.round(np.mean(degrees),4))
						i+=1
				else:
					continue
			else:
				continue
		# Start to compute normalized cut, volumes, cut and runtimes relative to the graph for each method
		# GAP
		nc_gap,vols_best,part,cut_g,t1_best=best_part(h,graph,n_iter)
		vola_best,volb_best=vols_best[0],vols_best[1]
		# Approximated Spectral
		nc_eig,volumes_eig,part_spectral,predictions_eig,cut_e,t1_eig=best_part_eig(f,g,graph,n_iter)
		vola_eig,volb_eig=volumes_eig[0],volumes_eig[1]

		# Spectral
		t0_true=timeit.default_timer()				
		L=laplacian(graph).to(device)
		L=L.coalesce().cpu()
		L_scipy=coo_matrix((L.values().numpy(),(L.indices()[0].numpy(),L.indices()[1].numpy())))
		first_eigs,true_eigenv=scipy.sparse.linalg.eigs(L_scipy,k=2,which='SR')
		first_eigenv=np.real(true_eigenv)
		t1_true=timeit.default_timer()-t0_true
		first_eigs=np.real(first_eigs)
		part_spectral_true,nc_eig_true,vols_e_true,cut_t=nc_eig_median(first_eigenv[:,1],g)
		vola_e_true,volb_e_true=vols_e_true[0],vols_e_true[1]
		
		# METIS
		nc_metis,t1_met,vola_m,volb_m,cut_m=normalized_cut_metis(g)
		 
		# SCOTCH
		nc_scotch,t_scotch,vola_scotch,volb_scotch,cut_s=scotch_partition(graph)

		# If GAP fails then we exclude this graph and we record it
		if nc_gap==2:
			print('GAP fails on this graph')
			print('')
			if dataset_type!='delaunay':
				names_failed.append(str(matrix))
			else:
				count_failed+=1
			continue
			
		# Collect all the produced data: number of nodes and edges, normalized cut, volumes, cut and runtimes for all the methods
		nodes_graphs.append(graph.num_nodes)
		edges_graphs.append(graph.num_edges)

		vols.append(max([vola_best/volb_best,volb_best/vola_best]))
		vols_eig.append(max([vola_eig/volb_eig,volb_eig/vola_eig]))
		vols_met.append(max([vola_m/volb_m,volb_m/vola_m]))
		vols_eig_true.append(max([vola_e_true/volb_e_true,volb_e_true/vola_e_true]))
		vols_scotch.append(max([vola_scotch/volb_scotch,volb_scotch/vola_scotch]))
		
		nc_gap_single.append(nc_gap)
		nc_metis_single.append(nc_metis)
		nc_scotch_single.append(nc_scotch)
		nc_eig_single.append(nc_eig)
		nc_true_single.append(nc_eig_true)
		
		c_gap_single.append(cut_g)
		c_metis_single.append(cut_m)
		c_scotch_single.append(cut_s)
		c_eig_single.append(cut_e)
		c_true_single.append(cut_t)
		
		t_gap_single.append(t1_best)
		t_metis_single.append(t1_met)
		t_scotch_single.append(t_scotch)
		t_eig_single.append(t1_eig)
		t_true_single.append(t1_true)

		# Print all the data for this graph
		print('Normalized cut:  GAP:',np.round(nc_gap,4),'  App. Spectral:',np.round(nc_eig,4),'  METIS:',np.round(nc_metis,4),'  Spectral:',np.round(nc_eig_true,4),'  Scotch:',np.round(nc_scotch,4))
		print('Balance:  GAP:',np.round(max([vola_best/volb_best,volb_best/vola_best]),4),'  App. Spectral:',np.round(max([vola_e_true/volb_e_true,volb_e_true/vola_e_true]),4),'  METIS:',np.round(max([vola_m/volb_m,volb_m/vola_m]),4),'  Spectral:',np.round(max([vola_eig/volb_eig,volb_eig/vola_eig]),4),'  Scotch:',np.round(max([vola_scotch/volb_scotch,volb_scotch/vola_scotch]),4))
		print('Cut:  GAP:',np.round(cut_g,4),'  App. Spectral:',np.round(cut_e,4),'  METIS:',np.round(cut_m,4),'  Spectral:',np.round(cut_t,4),'  Scotch:',np.round(cut_s,4))
		print('Time (seconds):  GAP:',np.round(t1_best,4),'  App. Spectral:',np.round(t1_eig,4),'  METIS:',np.round(t1_met,4),'  Spectral:',np.round(t1_true,4),'  Scotch:',np.round(t_scotch,4))
		print('')
		
	# Print the median for all the recorded quantities for each method
	if nodes_graphs!=[]:
		print('')
		print('Number of graphs:',i)
		print('Max nodes:',np.max(nodes_graphs),'  Max edges:',np.max(edges_graphs))
		print('Median normalized cut:  GAP:',np.round(np.median(nc_gap_single),4),'  App. Spectral:',np.round(np.median(nc_eig_single),4),'  METIS:',np.round(np.median(nc_metis_single),4),'  Spectral:',np.round(np.median(nc_true_single),4),'  Scotch:',np.round(np.median(nc_scotch_single),4))
		print('Median balance:  GAP:',np.round(np.median(vols),4),'  App. Spectral:',np.round(np.median(vols_eig),4),'  METIS:',np.round(np.median(vols_met),4),'  Spectral:',np.round(np.median(vols_eig_true),4),'  Scotch:',np.round(np.median(vols_scotch),4))
		print('Median cut:  GAP:',np.round(np.median(c_gap_single),4),'  App. Spectral:',np.round(np.median(c_eig_single),4),'  METIS:',np.round(np.median(c_metis_single),4),'  Spectral:',np.round(np.median(c_true_single),4),'  Scotch:',np.round(np.median(c_scotch_single),4))
		print('Median runtime:  GAP:',np.round(np.median(t_gap_single),4),'  App. Spectral:',np.round(np.median(t_eig_single),4),'  METIS:',np.round(np.median(t_metis_single),4),'  Spectral:',np.round(np.median(t_true_single),4),'  Scotch:',np.round(np.median(t_scotch_single),4))
		if dataset_type!='delaunay':
			print('Graphs for which GAP fails:',names_failed)
		else:
			print('Number of graphs for which GAP fails:',count_failed)
		print('')

