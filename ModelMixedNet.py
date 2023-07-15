import os
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


class base_layer(nn.Module):
    def __init__(self, hidden1, hidden2, batchnorm='none',dropout=0.5):
        super(base_layer, self).__init__()
        self.linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU())
        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(hidden2)
    def forward(self,feature):
        output = self.linear_layer(feature)
        if self.bn:
            output=self.bn_layer(output)
        return output


class Tower(nn.Module):
    def __init__(self, feature_size_des, feature_size_fp,
                 hidden_des=None, hidden_fp=None, batchnorm=None, dropout=None):
        super(Tower, self).__init__()

        if hidden_des is None:
           hidden_des = [64]
        if hidden_fp is None:
            hidden_fp = [64]

        self.hidden_des = hidden_des
        self.hidden_fp = hidden_fp

        n_layers = len(hidden_des)
        # self.hidden_S = hidden
        if dropout is None:
            dropout = [0.5 for _ in range(n_layers)]
        if batchnorm is None:
            batchnorm = [True for _ in range(n_layers)]
        lengths = [len(hidden_des), len(hidden_fp), len(batchnorm), len(dropout)]
        assert len(set(lengths)) == 1

        self.TowerS_layers_des = nn.ModuleList()
        for i in range(n_layers):
            self.TowerS_layers_des.append(base_layer(feature_size_des, hidden_des[i],dropout[i]))
            feature_size_des = hidden_des[i]

        self.TowerS_layers_fp = nn.ModuleList()
        for i in range(n_layers):
            self.TowerS_layers_fp.append(base_layer(feature_size_fp, hidden_fp[i], dropout[i]))
            feature_size_fp = hidden_fp[i]

    def forward(self, des_D,fp_D):
        for layer_des in self.TowerS_layers_des:
            des_D = layer_des(des_D)
        for layer_fp in self.TowerS_layers_fp:
            fp_D = layer_fp(fp_D)
        feature = torch.cat([des_D, fp_D], dim=1)
        return feature


class MixNet(nn.Module):
    def __init__(self,feature_size_des, feature_size_fp, hidden_shared, hidden, output, tasks_num,
                 hidden_des=None, hidden_fp=None, batchnorm_tower=None, dropout_tower=None,
                 dropout=0.5):
        super(MixNet, self).__init__()
        self.tower1 = clones(Tower(feature_size_des, feature_size_fp,
                            hidden_des= hidden_des, hidden_fp=hidden_fp,
                            batchnorm=batchnorm_tower, dropout=dropout_tower), tasks_num)

        in_feats = self.tower1[0].hidden_des[-1] + self.tower1[0].hidden_fp[-1]
        self.shared_layer = nn.Sequential(nn.Dropout(dropout),
                                          nn.Linear(in_feats, hidden_shared),
                                          nn.ReLU(),
                                          nn.BatchNorm1d(hidden_shared),
                                          # nn.Dropout(dropout),
                                          # nn.Linear(hidden_shared, hidden),
                                          # nn.ReLU(),
                                          # nn.BatchNorm1d(hidden),
                                          )
        self.output_layer = clones(nn.Sequential(
                                          nn.Linear(hidden_shared,output)),tasks_num)

    def mixnet_predict(self, descriptiors, fingerprints,tasks_num):
        outs = []
        for i in range(tasks_num):
            locals()['output'+str(i)] = self.tower1[i](descriptiors[i],fingerprints[i])
            locals()['output'+str(i)] = self.shared_layer(locals()['output'+str(i)])
            locals()['output'+str(i)] = self.output_layer[i](locals()['output'+str(i)])
            outs.append(locals()['output'+str(i)])
        return outs

