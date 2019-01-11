import argparse
import os
from copy import deepcopy
from operator import itemgetter
from heapq import nsmallest

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from omniglotNShot import OmniglotNShot

class Learner(nn.Module):

    def __init__(self, config, imgc, imgsz, device):

        super(Learner, self).__init__()
        
        self.config2vars = [None] * len(config)
        self.config2vars_bn = [None] * len(config)
        self.config = config
        self.device = device
        
        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()
        
        self.pruning_record = []
    
        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.config2vars[i] = len(self.vars) // 2 - 1

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))
                self.config2vars[i] = len(self.vars) // 2 - 1

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.config2vars[i] = len(self.vars) // 2 - 1

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))
                self.config2vars[i] = len(self.vars) // 2 - 1
                
                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])
                self.config2vars_bn[i] = len(self.vars_bn) // 2 - 1


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError


    def update(config,vars):
        self.config = config
        self.vars = vars

        return



    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True, hook=False, finetune=False):


        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        layer_index = 0### for pruning

        for layer,(name, param) in enumerate(self.config):
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]

                if finetune:
                    print(w.shape, b.shape)
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])

                idx += 2


            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]

                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2

            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2

            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':

                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError


        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return x

    def get_prune_plan(self, prune_number_one_epoch, max_prune_number):
        idx = 0
        weight_rank = []
        grad_rank = []
        
        idx =0
        for layer,(name, param) in enumerate(self.config):
            
            if name is 'conv2d':
                w = self.vars[idx]
                g_w = w.grad.cpu()
                values = g_w.data.flatten()
                
                grad_rank.extend(values)
                weight_rank.extend(w.detach().cpu().numpy().flatten())
            if name is 'relu':
                continue
            if name is 'flatten':
                continue
            idx += 2
            

        see_for_sure = []
        for i in range(len(weight_rank)):
            weight_rank[i] = abs(weight_rank[i])
            if weight_rank[i]==0:
                if i in self.pruning_record:
                    see_for_sure.append(i)
        result1 = np.argsort(np.array(weight_rank))[0:max_prune_number]
        print(len(see_for_sure))
        for i in range(len(grad_rank)):
            grad_rank[i] =  abs(grad_rank[i])
        result2 = np.argsort(np.array(grad_rank))[0:max_prune_number]
        
        max_count = 0
        result = []

        for i in result1:
            for j in result2:
                if(max_count >= prune_number_one_epoch):
                    break
                if i==j:
                    if i not in self.pruning_record:
                        result.append(i)
                        self.pruning_record.append(i)
                        max_count += 1
                        
        return result
    
    def zero_grad(self, vars=None):

        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):

        return self.vars

