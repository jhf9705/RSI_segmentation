import torch
import torch.nn.functional as F
from torch import nn
import os, sys
import torch.optim as optim
import numpy as np
import random
import cv2
import os
import matplotlib.image as mp
import torch.nn as nn
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

rate_of_unknown = 0.04  # 如未知类别像素点（黑色）超过这个比例，则忽略这张图片
dest = "G:\\0922\\5classes_small\\"  # 数据所在文件夹

# 开始网络搭建部分，这里使用第三方库smp来搭建，代码效率高很多
net = smp.Unet(
    encoder_name="resnet50",  # 选择主干网络
    encoder_weights="imagenet",  # use `imagenet` pretreined weights for encoder initialization
    in_channels=4,  # 4波段
    classes=6,  # 总分类数
)
net = net.cuda()  # 转移到GPU上
for_train = 10000  # 总计15000张图片，其中用于训练的张数，剩余的用作测试
print("net is built")
# 网络搭建部分结束


# 开始训练部分
l1 = []
for root, dirs, files in os.walk(dest, 'r'):  # 获取路径下所有文件
    for name in files:
        if ('_data.pth' in name):  # 表明此文件是训练数据
            l1.append(name)
random.shuffle(l1)  # 随机打乱列表
temp = np.array(l1)
np.save(dest + 'l1.npy', temp)  # 保存列表
l_train = l1[:for_train]  # 获取训练集
weight=torch.FloatTensor([1,0.7,0.7,2,4,2])
criterion = nn.CrossEntropyLoss(weight=weight,ignore_index=0).cuda()  # reduction='mean' #标签为0的是未知类，不计入损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001,betas=(0.9, 0.999))  #

batch_size = 1
inputs = torch.zeros(batch_size, 4, 672, 672).cuda()
la = torch.zeros(batch_size, 672, 672).cuda()
epoch = 20  # 迭代次数

for n in range(epoch):
    load = 0
    num = 0
    cor_all = 0
    pix_all=0
    random.shuffle(l_train)
    for i in l_train:
        img_path = dest + i
        lab_path = dest + i.replace('_data', '_label')  # 同一张图片，数据和标签在命名上只要将data换成label就可以

        ii = torch.load(img_path)[:, :672, :672]  # 因为图片是680*680的，但是网络要求尺寸必须是32的整数倍，因此取672*672，下同
        ll = torch.load(lab_path)[:672, :672]
        # ll[ll==0]=6
        # ll-=1


        yt = torch.where(ll == 0)
        length = len(yt[0])
        #print(torch.max(ll))
        # if (length > (672 * 672 * rate_of_unknown)):  # 统计未知类别像素点数目，超过一定比例则跳过这张图片
        #     continue



        inputs[load] = ii
        la[load] = ll
        load += 1

        if (load == batch_size):
            load = 0

            num += batch_size
            output = net(inputs)

            loss = criterion(output.squeeze(1), la.long().squeeze(1))  # 得到损失函数

            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 通过梯度做一步参数更新

            if (num % 100 == 0):
                outputs = output.cpu()
                lab = la.cpu()
                _, predicted = torch.max(outputs, 1)
                predicted = predicted.squeeze()
                correct = (predicted == lab).sum()

                is_unknown=(lab==0)  #标签图中为0的像素点
                ss=is_unknown.sum()  #统计像素点数目

                pix_all+=672*672*batch_size-ss  # 在统计正确率时忽略这些像素


                cor_all += correct
                if(pix_all.item()!=0):
                    print(str(n + 1) + ' ' + str(cor_all.item() /  pix_all.item()))  # 判断正确率
#    print(num)

# 训练部分结束，保存模型
save_path = dest + 'Unet_680_0926.pth'
torch.save(net, save_path)

# # 开始验证部分
# l_test = l1[for_train:]  # 验证集的文件列表
# num = 0
# cor_all = 0
# batch_size = 1
# net = net.cpu()
# inputs = torch.zeros(batch_size, 4, 672, 672)  # .cuda()
# la = torch.zeros(batch_size, 672, 672)  # .cuda()
#
# for i in l_test:
#     img_path = dest + i
#     lab_path = dest + i.replace('_data', '_label')
#
#     ii = torch.load(img_path)[:, :672, :672]
#     ll = torch.load(lab_path)[:672, :672]
#
#     yt = torch.where(ll == 0)
#     length = len(yt[0])
#     if (length > (672 * 672 * rate_of_unknown)):
#         continue
#
#     else:
#         inputs[load] = ii
#         la[load] = ll
#         load += 1
#
#     if (load == batch_size):
#         load = 0
#
#         num += batch_size
#         output = net(inputs)
#
#         outputs = output.cpu()
#         lab = la.cpu()
#         _, predicted = torch.max(outputs, 1)
#         predicted = predicted.squeeze()
#         correct = (predicted == lab).sum()
#         cor_all += correct
#         if (num % 10 == 0):
#             print(str(cor_all / num / 672 / 672))  # 输出总正确率
# # 验证部分结束

