# -*- coding: utf-8 -*- #
"""
@Project    ：MachineLearningLesson
@File       ：DataLoad.py
@Author     ：ZAY
@Time       ：2023/5/30 15:44
@Annotation : "自定义数据集 "
"""

import numpy as np
# import  pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, MinMaxScaler, Normalizer, StandardScaler

# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


BATCH_SIZE = 32
Test_Batch_Size = 99  # 32
random_state = 80


# 139, 208,256,  415, 484,553
# 自定义加载数据集
class MyDataset(Dataset):
    def __init__(self,specs,labels):
        self.specs = specs
        self.labels = labels

    def __getitem__(self, index):
        spec,target = self.specs[index],self.labels[index]
        return spec,target

    def __len__(self):
        return len(self.specs)

# 通过创建data.Dataset子类Mydataset来创建输入
class Mydataset(Dataset):
    # init() 初始化方法，传入数据文件夹路径
    def __init__(self, root):
        self.imgs_path = root

    # getitem() 切片方法，根据索引下标，获得相应的图片
    def __getitem__(self, index):
        img_path = self.imgs_path[index]

    # len() 计算长度方法，返回整个数据文件夹下所有文件的个数
    def __len__(self):
        return len(self.imgs_path)


class Mydatasetpro(Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    # 进行切片
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        pil_img = Image.open(img)  # pip install pillow
        pil_img = pil_img.convert('RGB')
        data = self.transforms(pil_img)
        return data, label

    def __getpath__(self, index):
        path = self.imgs[index]
        return path

    # 返回长度
    def __len__(self):
        return len(self.imgs)

