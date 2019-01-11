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
from learner import Learner
from omniglotNShot import OmniglotNShot


class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config, device):

        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = Learner(config, args.imgc, args.imgsz, device)
        self.device = device
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

        self.pruning_counter = 0
        

    def clip_grad_by_norm_(self, grad, max_norm):

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def forward(self, x_spt, y_spt, x_qry, y_qry, finetune=False):


        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i], i is tasks idx
        corrects = [0 for _ in range(self.update_step + 1)]


        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True, hook=False)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()

        self.meta_optim.step()
        if finetune:
            for weight_index in self.net.pruning_record:

                idx = 0
                number_counter = 0
                for layer,(name, param) in enumerate(self.net.config):
                    if name is 'conv2d':
                        tmp_counter = number_counter + param[0]*param[1]*param[2]*param[3]
                        if(weight_index >= number_counter and weight_index < tmp_counter):
                            l0 = int((weight_index - number_counter) / (param[1]*param[2]*param[3]))
                            l1 = int((weight_index - number_counter - l0*param[1]*param[2]*param[3]) / (param[2]*param[3]))
                            l2 = int((weight_index - number_counter - l0*param[1]*param[2]*param[3] - l1*param[2]*param[3]) / (param[3]))
                            l3 = int(weight_index - number_counter - l0*param[1]*param[2]*param[3] - l1*param[2]*param[3] - l2*param[3])
                            w = self.net.vars[idx].detach()
                            w[l0][l1][l2][l3] = 0
                            self.net.vars[idx] = torch.nn.Parameter(w)
                        number_counter += param[0]*param[1]*param[2]*param[3]
                    if name is 'relu':
                        continue
                    if name is 'flatten':
                        continue
                    idx += 2


        accs = np.array(corrects) / (querysz * task_num)

        return accs
    #########funcs for pruning##########################
    def train_epoch(self,x_spt, y_spt, x_qry, y_qry):
        self.net.zero_grad()

        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        loss_sum = 0


        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True, hook=True)
            loss = F.cross_entropy(logits, y_spt[i])
            loss_sum += loss
#    

        loss_q = loss_sum / task_num
        self.meta_optim.zero_grad()
        # optimize theta parameters
        loss_q.backward()
    #prune conv filter one by one
    def prune_conv_layer(self, weight_index):
        
        #config = self.net.config
        #name, para_list = config[original_layer_index]
        vars = self.net.vars
        
        idx = 0
        number_counter = 0
        for layer,(name, param) in enumerate(self.net.config):
            if name is 'conv2d':
                tmp_counter = number_counter + param[0]*param[1]*param[2]*param[3]
                
                if(weight_index >= number_counter and weight_index < tmp_counter):
                    l0 = int((weight_index - number_counter) / (param[1]*param[2]*param[3]))
                    l1 = int((weight_index - number_counter - l0*param[1]*param[2]*param[3]) / (param[2]*param[3]))
                    l2 = int((weight_index - number_counter - l0*param[1]*param[2]*param[3] - l1*param[2]*param[3]) / (param[3]))
                    l3 = int(weight_index - number_counter - l0*param[1]*param[2]*param[3] - l1*param[2]*param[3] - l2*param[3])
                    
                    weight = vars[idx].detach()

                    weight[l0][l1][l2][l3] = 0
                    
                    
                number_counter += param[0]*param[1]*param[2]*param[3]

            if name is 'relu':
                continue
            if name is 'flatten':
                continue
            idx += 2


    #calculate the total number of filters
    def filter_number(self):
        f = 0
        for name,module in self.net.config:
            if name=='conv2d':
                f += module[0]*module[1]*module[2]*module[3]
        return f

    


    def get_prune_index(self,x_spt, y_spt, x_qry, y_qry, prune_number_one_epoch, max_prune_number):
        #self.net.reset_prune()
        self.train_epoch(x_spt, y_spt, x_qry, y_qry)
#         self.net.layer_normalize()

        return self.net.get_prune_plan(prune_number_one_epoch, max_prune_number)

    #main func to prune
    def prune(self,x_spt, y_spt, x_qry, y_qry, prune_number_one_epoch, max_prune_number):
        total_filters = self.filter_number()
        

        # print("iterations:", iteration)

        # for _ in range(iteration):
        print("ranking")
        prune_index = self.get_prune_index(x_spt, y_spt, x_qry, y_qry, prune_number_one_epoch, max_prune_number)
        self.pruning_counter += len(prune_index)
        print("pruning filter...")

        for weight_index in prune_index:
            self.prune_conv_layer(weight_index)
        # message = str(100*float(self.net.filter_number()) / total_filters) + "%"
        # print("Filters prunned,there are left", message)
        # #self.test()
        # print("fine tuning..")


    ###################################################
    def finetunning(self, x_spt, y_spt, x_qry, y_qry):

        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz

        return accs
