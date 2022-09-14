# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:25:53 2022

@author: user
"""

import networkx as nx
import pandas as pd
from sklearn import preprocessing
import bamt.Preprocessors as pp
from fedot.core.pipelines.convert import graph_structure_as_nx_graph
from pgmpy.models import BayesianModel
from pgmpy.estimators import K2Score, BicScore, BDeuScore
import sys

parentdir = 'C:\\Users\\user\\Downloads\\FEDOT-fedot_for_bn'
sys.path.insert(0, parentdir)

def get_parents(graph, vertix):
    parents = nx.ancestors(graph, vertix)
    vertices = list(vertix.split()*len(parents))
    return list(zip(parents, vertices))

def get_childes(graph, vertix):
    childes = nx.descendants(graph, vertix)
    vertices = list(vertix.split()*len(childes))
    return list(zip(vertices, childes))

def get_childes_reversed(graph, vertix):
    childes = nx.descendants(graph, vertix)
    vertices = list(vertix.split()*len(childes))
    return list(zip(childes, vertices))

def custom_metric(graph, data: pd.DataFrame, method = 'K2'):
    score = 0
    nodes = data.columns.to_list()
    #graph_nx, labels = graph_structure_as_nx_graph(graph)
    struct = graph.edges
    #for pair in graph_nx.edges():
        #l1 = str(labels[pair[0]])
        #l2 = str(labels[pair[1]])
        #struct.append([l1, l2])
    bn_model = BayesianModel(struct)
    no_nodes = []
    for node in nodes:
        if node not in bn_model.nodes():
            no_nodes.append(node)
    if method == 'K2':
        score = K2Score(data).score(bn_model)
    elif method == 'Bic':
        score = BicScore(data).score(bn_model)
    elif method == 'BDeu':
        score = BDeuScore(data).score(bn_model)
    else:
        print('No such method. Select K2, Bic or BDeu')
    return [score]

def _RandomDAG_score(met):

    data=pd.read_csv(f'{parentdir}/examples/data/Cluster_2.csv', delimiter=',',decimal='.')
    vertices = ['Well', 'Layer', 'Top',
                'Bot', 'Gross', 'Saturation', 'NetPay',
                'Porosity', 'Permeability', 'Water Saturation Irreducible']
    data = data[vertices]
    encoder = preprocessing.LabelEncoder()
    discretizer = preprocessing.KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    p = pp.Preprocessor([('encoder', encoder), ('discretizer', discretizer)])
    discretized_data, est = p.apply(data)
    
    G = nx.gnp_random_graph(10, 0.2, directed=True)
    vertices = ['Well', 'Layer', 'Top',
            'Bot', 'Gross', 'Saturation', 'NetPay',
            'Porosity', 'Permeability', 'Water Saturation Irreducible']
    mapping = dict(zip(G.nodes,vertices))
    G = nx.relabel_nodes(G, mapping)
    
    if nx.is_directed_acyclic_graph(G)==False:
        for vert in vertices:
            cross = list(set(get_parents(G,vert)) & set(get_childes_reversed(G, vert)))
            #print(cross)
            for wrong_edge in cross:
                if wrong_edge in G.edges:
                    #print(wrong_edge)
                    G.remove_edge(wrong_edge[0], wrong_edge[1])
                
#print(G.edges)

    #met='BDeu'
    OF=round(custom_metric(G, method=met, data=discretized_data)[0],2)                
    print(OF)

    #if nx.is_directed_acyclic_graph(G)==True:
        #print('DAG')
        #nx.draw_networkx(G, with_labels = True)
    
    return [OF, G.nodes, G.edges]

score_table= []
total = []
met='BDeu'
for i in range(1,11):
    #_RandomDAG_score(met)
    score_table.append(_RandomDAG_score(met))
    #total.append(score_table)
df = pd.DataFrame(score_table)
df.to_csv('Random graph score.csv', index=False)
 





