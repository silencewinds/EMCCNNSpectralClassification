# -*- coding: utf-8 -*-
# @Time    : 2019/2/13 11:00
# @Author  : HuangHao
# @Email   : 812116298@qq.com
# @File    : test.py
import os
import torch
import torch.utils.data as Data
from scipy.io import loadmat
import torch.backends.cudnn as cudnn

from models.networks import *
from data_prepare import *


def model_three_to_one(p1, p2, p3):
    p1 = p1.numpy()
    p2 = p2.numpy()
    p3 = p3.numpy()
    result_list = []
    for i in range(len(p1)):
        if p1[i]==p2[i]:
            result = p1[i]
        elif p1[i]==p3[i]:
            result = p1[i]
        elif p2[i]==p3[i]:
            result = p2[i]
        else:
            result = p1[i]
        result_list.append(result)
    result = torch.Tensor(np.array(result_list))

    return result



if __name__ == '__main__':
    spectraldata = SpectralData()
    train_dataset, test_dataset = spectraldata.dataPrepare_1d()  # prepare data
    testloader = Data.DataLoader(dataset=test_dataset,
                                 batch_size=64,
                                 shuffle=True,
                                 num_workers=1)
    print('==> Building model..')
    net1 = MyCNN()
    net2 = VGG('VGG16')
    net3 = MyCNNNB()
    model1 = torch.load(r'F:\PythonProject\SpectralClassification\experiment\mycnn 5-10\380.pth')
    model2 = torch.load(r'F:\PythonProject\SpectralClassification\experiment\vgg16 5-10\400.pth')
    model3 = torch.load(r'F:\PythonProject\SpectralClassification\experiment\mycnn 5-10 no bn\400.pth')



    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net1.cuda()
        net1 = torch.nn.DataParallel(net1, device_ids=[0])
        net1.load_state_dict(model1)
        net2.cuda()
        net2 = torch.nn.DataParallel(net2, device_ids=[0])
        net2.load_state_dict(model2)
        net3.cuda()
        net3 = torch.nn.DataParallel(net3, device_ids=[0])
        net3.load_state_dict(model3)

        cudnn.benchmark = True

    test_acc_list = []
    correct_epoch_avg = 0
    correct_epoch = 0
    correct_epoch1 = 0
    correct_epoch2 = 0
    correct_epoch3 = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs1 = net1(inputs)
        outputs2 = net2(inputs)
        outputs3 = net3(inputs)
        outputs = (outputs1 + outputs2 + outputs3)/3
        # 数据统计
        _, predicted_avg = torch.max(outputs.data, 1)
        _, predicted1 = torch.max(outputs1.data, 1)
        _, predicted2 = torch.max(outputs2.data, 1)
        _, predicted3 = torch.max(outputs3.data, 1)

        predicted1 = predicted1.type(torch.FloatTensor)
        predicted2 = predicted2.type(torch.FloatTensor)
        predicted3 = predicted3.type(torch.FloatTensor)
        predicted_avg = predicted_avg.type(torch.FloatTensor)
        predicted = model_three_to_one(predicted1, predicted2, predicted3)

        targets = torch.Tensor([np.argmax(one_hot) for one_hot in targets.data.cpu().numpy()])

        correct_avg = (predicted_avg == targets).sum()
        correct_avg = correct_avg.item() / float(64)
        correct_epoch_avg += correct_avg


        correct = (predicted == targets).sum()
        correct = correct.item() / float(64)
        correct_epoch += correct
        batchs = batch_idx + 1

        correct1 = (predicted1 == targets).sum()
        correct1 = correct1.item() / float(64)
        correct_epoch1 += correct1

        correct2 = (predicted2 == targets).sum()
        correct2 = correct2.item() / float(64)
        correct_epoch2 += correct2

        correct3 = (predicted3 == targets).sum()
        correct3 = correct3.item() / float(64)
        correct_epoch3 += correct3


    correct_epoch /= batchs
    correct_epoch1 /= batchs
    correct_epoch2 /= batchs
    correct_epoch3 /= batchs
    correct_epoch_avg /= batchs
    test_acc_list.append(correct_epoch)
    print('test correct:' + str(correct_epoch)+str(correct_epoch1)+str(correct_epoch2)+str(correct_epoch3)+str(correct_epoch_avg))
