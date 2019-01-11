import argparse
import os
from copy import deepcopy
from operator import itemgetter
from heapq import nsmallest

from meta import Meta
import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from omniglotNShot import OmniglotNShot

def main(args):

    config = [
        ('conv2d', [64, 1, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 2, 2, 1, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        ('linear', [args.n_way, 64])
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    maml = Meta(args, config, device).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))


    print('Total trainable tensors:', num)

    db_train = OmniglotNShot('./',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz)

    for step in range(args.epoch):

        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).long().to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).long().to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = maml(x_spt, y_spt, x_qry, y_qry)
        print('trainstep:', step, '\ttraining acc:', accs)

        if (step+1) % 500 == 0:
            accs = []
            for _ in range(1000//args.task_num):
                # test
                x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).long().to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).long().to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)

    ##############################
    for i in range(args.prune_iteration):
        # prune
        print("the {}th prune step".format(i))
        x_spt, y_spt, x_qry, y_qry = db_train.getHoleTrain()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).long().to(device), \
                                 torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).long().to(device)
        maml.prune(x_spt, y_spt, x_qry, y_qry, args.prune_number_one_epoch, args.max_prune_number)




        # fine-tuning
        print("start finetuning....")
        finetune_epoch = args.finetune_epoch
        finetune_epoch = finetune_epoch* (2 if i == args.prune_iteration - 1 else 1)

        for step in range(args.finetune_epoch):
            x_spt, y_spt, x_qry, y_qry = db_train.next()
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).long().to(device), \
                                         torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).long().to(device)
            accs = maml(x_spt, y_spt, x_qry, y_qry, finetune=True)

            print('finetune step:', step, '\ttraining acc:', accs)

        # print the test accuracy after pruning
        print("start testing....")
        accs = []
        for _ in range(1000//args.task_num):
            # test
            x_spt, y_spt, x_qry, y_qry = db_train.next('test')
            x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).long().to(device), \
                                         torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).long().to(device)

            # split to single task each time
            for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                test_acc = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                accs.append( test_acc )

        # [b, update_step+1]
        accs = np.array(accs).mean(axis=0).astype(np.float16)
        print('Test acc:', accs)


if __name__ == '__main__':


    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=100000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.6)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    argparser.add_argument('--prune_iteration', type=int, help='number of pruning epochs', default=5)
    argparser.add_argument('--finetune_epoch', type=int, help='number of finetuning epochs after each pruning epoch', default=50)
    argparser.add_argument('--prune_number_one_epoch', type=int, help='number of neurons to prune in each epoch', default=1000)
    argparser.add_argument('--max_prune_number', type=int, help='max number of nuerons to be pruned', default=60000)

    args = argparser.parse_args()

    main(args)
