# -*- coding: utf-8 -*-
# @Time    : 2019/2/1 11:22
# @Author  : HuangHao
# @Email   : 812116298@qq.com
# @File    : utils.py

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']

def pltline(time, train_acc_list, test_acc_list, loss_list):
    plt.figure(figsize=(10,5))
    plt.suptitle('experiment result', fontsize=13, x=0.5, y=1)
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_list, '--',  label='training accuracy', color='black')
    plt.plot(test_acc_list,  label='test accuracy', color='black')
    plt.ylim(0,1)
    plt.legend()  # 显示图例
    plt.xlabel('epoch')
    plt.ylabel('rate')
    plt.subplot(1, 2, 2)
    plt.plot(loss_list, '-.', label='loss', color='black')
    plt.legend()  # 显示图例
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.savefig('./experiment/{0}/result.png'.format(time))
    plt.show()

def plotImg():
    m0_5 = loadmat('F:\PythonProject\SpectralClassification\dataset\M0_5_10_5000.mat')['P1']
    m0_15 = loadmat('F:\PythonProject\SpectralClassification\dataset\M0_15_5000.mat')['P1']
    plt.figure(figsize=(10, 5))
    plt.suptitle('不同信噪比的光谱对比图', fontsize=13, x=0.5, y=1)
    plt.subplot(1, 2, 1)
    plt.ylabel("信噪比5-10")
    plt.plot(m0_5[1],color='black')

    plt.subplot(1, 2, 2)
    plt.ylabel("信噪比大于15")
    plt.plot(m0_15[5],color='black')
    plt.show()

if __name__ == '__main__':
    plotImg()
# if __name__ == '__main__':
#     pltline('1', [55,66,77,88], [44,55,66,77], [3,2,1,0.5])