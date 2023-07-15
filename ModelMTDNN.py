import os
import pandas as pd
import numpy as np
from torch import nn, optim
import dgl
import dgl.function as fn   #使用内置函数并行更新API
import torch.nn.functional as F
import copy
import itertools
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import math
import statistics

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 多任务DNN
class MTLnet(nn.Module):
    def __init__(self, feature_size, shared_layer_size, tower_h1, tower_h2, tasks_num, output_size,dropout = 0.5):
        super(MTLnet, self).__init__()

        self.sharedlayer = nn.Sequential(
            nn.Linear(feature_size, shared_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=shared_layer_size),
            nn.Dropout(dropout)
            )
        self.tower = clones(nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tower_h2, output_size)), tasks_num)
    def forward(self, x, tasks_num):
        outs = []
        for i in range(tasks_num):
            locals()['h_shared' + str(i+1)]= self.sharedlayer(x[i])
            locals()['out' + str(i + 1)] = self.tower[i](locals()['h_shared' + str(i+1)])
            # locals()['out' + str(i + 1)] = F.softmax(locals()['out' + str(i + 1)], dim=1)
            outs.append(locals()['out' + str(i + 1)])
        return outs

#############################################################################################################################
class MTLnet_2Layer(nn.Module):
    def __init__(self, feature_size, shared_layer_size, tower_h1, tasks_num, output_size,dropout = 0.5):
        super(MTLnet_2Layer, self).__init__()

        self.sharedlayer = nn.Sequential(
            nn.Linear(feature_size, shared_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=shared_layer_size),
            nn.Dropout(dropout)
            )
        self.tower = clones(nn.Sequential(
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(tower_h1, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(tower_h1, output_size)), tasks_num)
    def forward(self, x, tasks_num):
        outs = []
        for i in range(tasks_num):
            locals()['h_shared' + str(i+1)]= self.sharedlayer(x[i])
            locals()['out' + str(i + 1)] = self.tower[i](locals()['h_shared' + str(i+1)])
            # locals()['out' + str(i + 1)] = F.softmax(locals()['out' + str(i + 1)], dim=1)
            outs.append(locals()['out' + str(i + 1)])
        return outs

#############################################################################################################################

class MTLnet_1Layer(nn.Module):
    def __init__(self, feature_size, shared_layer_size, tasks_num, output_size,dropout = 0.5):
        super(MTLnet_1Layer, self).__init__()

        self.sharedlayer = nn.Sequential(
            nn.Linear(feature_size, shared_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=shared_layer_size),
            nn.Dropout(dropout)
            )
        self.tower = clones(nn.Sequential(
            nn.Linear(shared_layer_size, output_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(tower_h1, tower_h2),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            # nn.Linear(tower_h1, output_size)
        ), tasks_num)
    def forward(self, x, tasks_num):
        outs = []
        for i in range(tasks_num):
            locals()['h_shared' + str(i+1)]= self.sharedlayer(x[i])
            locals()['out' + str(i + 1)] = self.tower[i](locals()['h_shared' + str(i+1)])
            # locals()['out' + str(i + 1)] = F.softmax(locals()['out' + str(i + 1)], dim=1)
            outs.append(locals()['out' + str(i + 1)])
        return outs
################################################################################################################
class MTLnet_2shared(nn.Module):
    def __init__(self, feature_size, shared_layer_size, tower_h1, tower_h2, tasks_num, output_size,dropout = 0.5):
        super(MTLnet_2shared, self).__init__()

        self.sharedlayer = nn.Sequential(
            nn.Linear(feature_size, shared_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=shared_layer_size),
            nn.Dropout(dropout),
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(dropout)
            )
        self.tower = clones(nn.Sequential(
            # nn.Linear(feature_size, shared_layer_size),
            # nn.ReLU(),
            # nn.BatchNorm1d(num_features=shared_layer_size),
            # nn.Dropout(dropout),
            # nn.Linear(shared_layer_size, tower_h1),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tower_h2, output_size)), tasks_num)
    def forward(self, x, tasks_num):
        outs = []
        for i in range(tasks_num):
            # locals()['out' + str(i + 1)] = self.tower[i](x[i])   # MST
            locals()['h_shared' + str(i+1)]= self.sharedlayer(x[i])
            locals()['out' + str(i + 1)] = self.tower[i](locals()['h_shared' + str(i+1)])
            # locals()['out' + str(i + 1)] = F.softmax(locals()['out' + str(i + 1)], dim=1)
            outs.append(locals()['out' + str(i + 1)])
        return outs

######################################################################################################################
class MTLnet_3shared(nn.Module):
    def __init__(self, feature_size, shared_layer_size, tower_h1, tower_h2, tasks_num, output_size, dropout=0.5):
        super(MTLnet_3shared, self).__init__()

        self.sharedlayer = nn.Sequential(
            nn.Linear(feature_size, shared_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=shared_layer_size),
            nn.Dropout(dropout),
            nn.Linear(shared_layer_size, tower_h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tower_h1, tower_h2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.tower = clones(nn.Sequential(
            nn.Linear(tower_h2, output_size)), tasks_num)

    def forward(self, x, tasks_num):
        outs = []
        for i in range(tasks_num):
            locals()['h_shared' + str(i + 1)] = self.sharedlayer(x[i])
            locals()['out' + str(i + 1)] = self.tower[i](locals()['h_shared' + str(i + 1)])
            # locals()['out' + str(i + 1)] = F.softmax(locals()['out' + str(i + 1)], dim=1)
            outs.append(locals()['out' + str(i + 1)])
        return outs