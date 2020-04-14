# -*- coding: utf-8 -*-
# @Time    : 2019/2/1 10:58
# @Author  : HuangHao
# @Email   : 812116298@qq.com
# @File    : train.py

'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as Data

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from scipy.io import loadmat
import numpy as np
import json
import time
import matplotlib.pyplot as plt

from models.networks import *
from data_prepare import *
from utils import pltline
from torch.autograd import Variable



# 获取参数
f = open('conf/setting.json')
json_args = json.load(f)
parser = argparse.ArgumentParser(description='PyTorch Spectral Classification Training')
parser.add_argument('--name', default=json_args['name'], help='net name')
parser.add_argument('--lr', default=json_args['lr'], type=float, help='learning rate')
parser.add_argument('-batchsize', default=json_args['batchsize'], help='batchsize')
parser.add_argument('-istrain', default=json_args['is_train'], help='is tarin')
parser.add_argument('-epoch', default=json_args['epoch'], help='max epoch')
parser.add_argument('-tf', default=json_args['tf'], help='test freq')
parser.add_argument('-savepoint', default=json_args['savepoint'], help='save point')
parser.add_argument('-dataset', default=json_args['dataset'], help='dateset path')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

best_acc = 0     # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch




def train(epoch, time):
    # switch to train mode
    net.train()                  #进入到训练模式
    train_loss = 0
    correct = 0
    total = 0
    correct_epoch = 0
    batchs = 0
    # batch 数据
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # 将数据移到GPU上
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # 先将optimizer梯度先置为0
        optimizer.zero_grad()
        # Variable表示该变量属于计算图的一部分，此处是图计算的开始处。图的leaf variable
        inputs, targets = Variable(inputs), Variable(targets)
        # 模型输出
        outputs = net(inputs)
        # 计算loss，图的终点处
        # outputs = outputs.LongTensor()
        # targets = targets.LongTensor()
        loss = criterion(outputs, targets)
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # 注意如果你想统计loss，切勿直接使用loss相加，而是使用loss.data[0]。因为loss是计算图的一部分，如果你直接加loss，代表total loss同样属于模型一部分，那么图就越来越大
        train_loss += loss.data.item()
        # 数据统计
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.type(torch.FloatTensor)
        total += targets.size(0)
        targets = torch.Tensor([np.argmax(one_hot) for one_hot in targets.data.cpu().numpy()])
        correct = (predicted==targets).sum()
        correct = correct.item()/float(args.batchsize)
        correct_epoch += correct
        batchs = batch_idx + 1

    correct_epoch /= batchs
    train_loss /= batchs
    print('epoch:'+str(epoch)+' '+'correct:'+str(correct_epoch)+" loss:"+str(train_loss))
    train_acc_list.append(correct_epoch)
    loss_list.append(train_loss)
    if epoch % args.tf == 0:
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            # 数据统计
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.type(torch.FloatTensor)
            total += targets.size(0)
            targets = torch.Tensor([np.argmax(one_hot) for one_hot in targets.data.cpu().numpy()])
            correct = (predicted == targets).sum()
            correct = correct.item() / float(args.batchsize)
            correct_epoch += correct
            batchs = batch_idx + 1

        correct_epoch /= batchs
        test_acc_list.append(correct_epoch)
        print('test correct:' + str(correct_epoch))

    if (epoch+1) % args.savepoint == 0:
        print("saving model...")
        torch.save(net.state_dict(), "./experiment/{0}/{1}.pth".format(time, str(epoch+1)))


if __name__ == '__main__':
    spectraldata = SpectralData()
    train_dataset, test_dataset = spectraldata.dataPrepare_1d()
    trainloader = Data.DataLoader(dataset=train_dataset,
                                  batch_size=args.batchsize,
                                  shuffle=True,
                                  num_workers=1)
    testloader = Data.DataLoader(dataset=test_dataset,
                                 batch_size=args.batchsize,
                                 shuffle=True,
                                 num_workers=1)
    print('==> Building model..')
    # net = VGG('VGG16')
     # print(net)
    # net = VGG2d('VGG16')
    # print(net)
    # net = MyDNN()
    # net = densenet121()
    net = MyCNN_fastpath_1d()
    print(args.name)

    # 如果GPU可用，使用GPU
    if use_cuda:
        # move param and buffer to GPU
        net.cuda()
        # parallel use GPU
        net = torch.nn.DataParallel(net, device_ids=[0])
        # speed up slightly
        cudnn.benchmark = True

    # 定义度量和优化
    criterion = nn.MultiLabelSoftMarginLoss()
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

    time = str(time.time())
    print("experiment/{0}".format(time))
    os.makedirs("experiment/{0}".format(time))

    train_acc_list = []
    test_acc_list = []
    loss_list = []

    for i in range(args.epoch):
        train(i, time)
    pltline(time, train_acc_list, test_acc_list, loss_list)
    print("finished")