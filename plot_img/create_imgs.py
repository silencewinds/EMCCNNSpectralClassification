# -*- coding: utf-8 -*-
# @Time    : 2019/3/14 19:21
# @Author  : HuangHao
# @Email   : 812116298@qq.com
# @File    : create_imgs.py
import matplotlib.pyplot as plt
import numpy as np
def plot_1_img():
    plt.figure(figsize=(10, 4))


    ax1 = plt.subplot(1, 2, 1)  # （行，列，活跃区）
    ax1.grid(True)
    y0 = [0.9057, 0.8579, 0.8804]
    y1 = [0.982, 0.9483, 0.9355]
    x = ['5-10','10-15','up 15']
    l1, = plt.plot(x, y0, 'y*-', label = 'line1',color='black')
    l2, = plt.plot(x, y1, 'yx-',label = 'line2',color='black')
    plt.xlabel('sn')
    plt.ylabel('train accuracy')
    plt.legend(handles = [l1, l2,], labels = ['dnn', 'cnn'], loc = 'best')


    ax1 = plt.subplot(1, 2, 2)  # （行，列，活跃区）
    ax1.grid(True)
    y0 = [0.8614,0.9076,0.9591]
    y1 = [0.9251,0.9706,0.7107]
    x = ['5-10', '10-15', 'up 15']
    l1, = plt.plot(x, y0, 'y*-', label='line1', color='black')
    l2, = plt.plot(x, y1, 'yx-', label='line2', color='black')
    plt.xlabel('sn')
    plt.ylabel('test accuracy')
    plt.legend(handles=[l1, l2, ], labels=['dnn', 'cnn'], loc='best')

    plt.suptitle('CNN and DNN classification of M star results', x=0.5, y=0.999)
    plt.show()

def plot_2_img():
    plt.figure(figsize=(10, 4))


    ax1 = plt.subplot(1, 2, 1)  # （行，列，活跃区）
    ax1.grid(True)
    y0 = [0.9768, 0.9513, 0.9322]
    y1 = [0.982, 0.9483, 0.9355]
    x = ['5-10','10-15','up 15']
    l1, = plt.plot(x, y0, 'y*-', label = 'line1',color='black')
    l2, = plt.plot(x, y1, 'yx-',label = 'line2',color='black')
    plt.xlabel('sn')
    plt.ylabel('train accuracy')
    plt.legend(handles = [l1, l2,], labels = ['dnn', 'cnn'], loc = 'best')


    ax1 = plt.subplot(1, 2, 2)  # （行，列，活跃区）
    ax1.grid(True)
    y0 = [0.9204, 0.9763, 0.9854]
    y1 = [0.9251, 0.9706, 0.7107]
    x = ['5-10', '10-15', 'up 15']
    l1, = plt.plot(x, y0, 'y*-', label='line1', color='black')
    l2, = plt.plot(x, y1, 'yx-', label='line2', color='black')
    plt.plot('up 15',0.9854,'o', markersize=15,color='black')
    plt.xlabel('sn')
    plt.ylabel('test accuracy')
    plt.legend(handles=[l1, l2, ], labels=['cnn no bn', 'cnn'], loc='best')

    plt.suptitle('Classification comparison of CNN network with or without batchnorm', x=0.5, y=0.999)
    plt.show()

def plot_3_img():
    plt.figure(figsize=(10, 4))


    ax1 = plt.subplot(1, 2, 1)  # （行，列，活跃区）
    ax1.grid(True)
    y0 = [0.982, 0.9483, 0.9355]
    y1 = [0.99, 0.9783, 0.9687]
    y2 = [0.9923, 0.9965, 0.999]
    x = ['5-10','10-15','up 15']
    l1, = plt.plot(x, y0, 'y*-', label = 'line1',color='black')
    l2, = plt.plot(x, y1, 'yx-',label = 'line2',color='black')
    l3, = plt.plot(x, y2, 'yo-',label = 'line3',color='black')
    plt.xlabel('sn')
    plt.ylabel('train accuracy')
    plt.legend(handles = [l1, l2, l3], labels = ['cnn', 'vgg16','res18'], loc = 'best')


    ax1 = plt.subplot(1, 2, 2)  # （行，列，活跃区）
    ax1.grid(True)
    y0 = [0.9251,0.9706,0.7107]
    y1 = [0.8905,0.9033,0.6687]
    y2 = [0.8632,0.8584,0.856]
    x = ['5-10', '10-15', 'up 15']
    l1, = plt.plot(x, y0, 'y*-', label='line1', color='black')
    l2, = plt.plot(x, y1, 'yx-', label='line2', color='black')
    l3, = plt.plot(x, y2, 'yo-', label='line3', color='black')
    plt.xlabel('sn')
    plt.ylabel('test accuracy')
    plt.legend(handles=[l1, l2, l3], labels=['cnn', 'vgg16', 'res18'], loc='best')

    plt.suptitle('CNN network, vgg16 network, res18 network classification comparison', x=0.5, y=0.999)
    plt.show()


if __name__ == '__main__':
    plot_3_img()