import torch

import tifffile as tiff
import numpy as np
import random

import matplotlib.pyplot as plt


for_train = 10000  # 总计15000张图片，其中用于训练的张数，剩余的用作测试
rate_of_unknown = 0.04  # 如未知类别像素点（黑色）超过这个比例，则忽略这张图片
dest = "G:\\0922\\5classes_small\\"  # 数据所在文件夹
dest_save="G:\\0922\\result\\"#存放生成图片文件夹


# 开始验证部分
l1=np.load(dest + 'l1.npy').tolist()  # 读取列表
save_path = dest + 'Unet_680_0924.pth'
net=torch.load(save_path)
l_test = l1[for_train:]  # 验证集的文件列表
random.shuffle(l_test)  # 随机打乱列表，如不想打乱可注释此行
num = 0
cor_all = 0
batch_size = 1
load=0
net = net.cuda()
inputs = torch.zeros(batch_size, 4, 672, 672).cuda()
la = torch.zeros(batch_size, 672, 672).cuda()
pix_all=0
for i in l_test:
    img_path = dest + i
    lab_path = dest + i.replace('_data', '_label')

    ii = torch.load(img_path)[:, :672, :672]
    ll = torch.load(lab_path)[:672, :672]

    yt = torch.where(ll == 0)
    length = len(yt[0])
    # if (length > (672 * 672 * rate_of_unknown)):
    #     continue


    inputs[load] = ii
    la[load] = ll
    load += 1

    if (load == batch_size):
        load = 0

        num += batch_size
        output = net(inputs)

        outputs = output.cpu()
        lab = la.cpu()
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.squeeze()
        correct = (predicted == lab).sum()

        is_unknown = (lab == 0)  # 标签图中为0的像素点
        ss = is_unknown.sum()  # 统计像素点数目

        pix_all += 672 * 672 * batch_size - ss  # 在统计正确率时忽略这些像素

        cor_all += correct

        predict_numpy=predicted.numpy()
        result=np.zeros((672,672,3),dtype=np.uint8)
        result[predict_numpy==1]=[255,0,0]
        result[predict_numpy == 2] = [255, 255, 0]
        result[predict_numpy == 3] = [0, 255, 0]
        result[predict_numpy == 4] = [127, 255, 0]
        result[predict_numpy == 5] = [0, 0, 255]



        lab_numpy=lab[0].numpy()
        label=np.zeros((672,672,3),dtype=np.uint8)
        label[lab_numpy==1]=[255,0,0]
        label[lab_numpy == 2] = [255, 255, 0]
        label[lab_numpy == 3] = [0, 255, 0]
        label[lab_numpy == 4] = [127, 255, 0]
        label[lab_numpy == 5] = [0, 0, 255]





        if (num % 50 == 0 and pix_all.item()!=0):
            print(str(cor_all.item()/ pix_all.item())) # 输出总正确率
            image=ii.numpy().transpose(1,2,0)
            tiff.imsave(dest_save+str(num)+"_image.tif",image)
            plt.imsave(dest_save+str(num)+"_predicted.png",result)
            plt.imsave(dest_save + str(num) + "_label.png", label)
            image=ii.numpy().transpose(1,2,0)
            tiff.imsave(dest_save+str(num)+"_image.tif",image)
            plt.imsave(dest_save+str(num)+"_predicted.png",result)
            plt.imsave(dest_save + str(num) + "_label.png", label)

# 验证部分结束

