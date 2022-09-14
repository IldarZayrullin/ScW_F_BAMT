import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = 'C:\\Users\\user\\Downloads\\FEDOT-fedot_for_bn'
sys.path.insert(0,parentdir)


from copy import copy
import math
from typing import List
import numpy as np
import pandas as pd
from bamt.external.pyBN.utils.independence_tests import mutual_information, entropy
from bamt.preprocess.discretization import get_nodes_type
from bamt.preprocess.numpy_pandas import loc_to_DataFrame
from bamt.preprocess.graph import edges_to_dict

def child_dict(net: list):
    res_dict = dict()
    for e0, e1 in net:
        #print(e0, e1)
        if e1 in res_dict:
            res_dict[e1].append(e0)
        else:
            res_dict[e1] = [e0]
    return res_dict

def SHD(pred_net: list, true_net: list, decimal = 2):
    pred_dict = child_dict(pred_net)
    true_dict = child_dict(true_net)
    corr_undir = 0
    corr_dir = 0
    for e0, e1 in pred_net:
        flag = True
        if e1 in true_dict:
            if e0 in true_dict[e1]:
                corr_undir += 1
                corr_dir += 1
                flag = False
        if (e0 in true_dict) and flag:
            if e1 in true_dict[e0]:
                corr_undir += 1
    pred_len = len(pred_net)
    true_len = len(true_net)
    shd = pred_len + true_len - corr_undir - corr_dir
    return shd
            #{'AP': round(corr_undir/pred_len, decimal), 'AR': round(corr_undir/true_len, decimal), 'AHP': round(corr_dir/pred_len, decimal), 'AHR': round(corr_dir/true_len, decimal), 'SHD': shd}


file_dir = 'C:\\Users\\user\\Downloads\\FEDOT-fedot_for_bn\\examples\\CrossMutation\\'
import os
files = os.listdir(file_dir)
#print(files)

#filename='Crossover 0.5 Mutation 0.5.csv'

import ast

def create_nodes_list(filename):
    
    graph_list = pd.read_csv(file_dir+filename,
                         skiprows=1, header = None, delimiter = ',', sep = '.,').squeeze('columns')
    file = graph_list.to_list()
    K2score_list = []
    total=[]
    vertex_list = []
    for i in range(0, len(graph_list)):
        part1 = file[i].split(',')[0]
        K2score_list.append(eval(part1[1:]))
        part3 = file[i].split('[')[2]
        vertexes = part3[:-5]
        vertex_list.append(vertexes.split(', '))
        part2 = file[i].split('"')[1]
        nodes_list = ast.literal_eval(part2)
        total.append(nodes_list)
        #print(nodes_list)
    return(K2score_list, vertex_list, total)
        
#K2_list, vertex_list, nodes_list = create_nodes_list(filename)
#print(nodes_list[0])

#pred_net=nodes_list[0]
#true_net=nodes_list[0]

#print(SHD(pred_net,true_net))

def score_matrix(nodes_list):
    n=len(nodes_list)
    #print(n)
    score_matrix=[]
    shd_max=0
    for i in range(0,n):
        row=[]
        for j in range(0,n):
            row.append(SHD(nodes_list[i],nodes_list[j])) 
            if SHD(nodes_list[i],nodes_list[j])>shd_max:
                shd_max=SHD(nodes_list[i],nodes_list[j])
        score_matrix.append(row)
    #print('Максимальное значение SHD = ', shd_max)
    return np.array(score_matrix), shd_max
    
#sc_m = score_matrix(nodes_list)
#print(sc_m.reshape(10,10))

import seaborn as sns
import matplotlib.pyplot as plt

#ax1 = sns.heatmap(sc_m, vmin=0, vmax=45)
#ax2 = sns.heatmap(sc_m)
#fig = ax2.get_figure()
#fig.savefig(file_dir+'heatmap '+filename[:-3]+'png') 

for filename in files:
    K2_list, vertex_list, nodes_list = create_nodes_list(filename)
#print(nodes_list)
#score_matrix(nodes_list)
    hist_data=[]
    sc_m, shd = score_matrix(nodes_list)
    large = 22; med = 16; small = 12
#print(sc_m)
    for i in range(0, len(sc_m)):
        hist_data.extend(sc_m[i][i+1:])

    #print(hist_data)
#sns.histplot(hist_data)
    plt.rcParams["figure.figsize"] = (10,10)
    plt.title('Гистограмма SHD для различных графов')
    params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
    plt.rcParams.update(params)
    plt.xlabel('Значение SHD')
    #plt.xlim = ([2, 24])
    #plt.ylim = ([0, 12])
    ax = plt.gca()
    ax.set_xlim([2, 24])
    ax.set_ylim([0, 12])
    plt.axvline(x=np.mean(hist_data), linewidth=4, color='r')
    plt.xticks(ticks = range(2,24), labels = range(2,24, 1))
    #plt.yticks(range(0, 12, 1))
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.hist(hist_data, bins = range(2,24), histtype='barstacked', facecolor='g', align = 'left')
    plt.savefig(file_dir+'hist '+filename[:-3]+'png')
    plt.clf()
    # print(shd)
    # ax1 = sns.heatmap(sc_m, vmin=0, vmax=25)
    # fig = ax1.get_figure()
    # fig.savefig(file_dir+'heatmap '+filename[:-3]+'png')
    # fig.clf()
