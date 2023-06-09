# -*- coding: utf-8 -*- #
"""
@Project    ：MachineLearningLesson
@File       ：Dataload.py 
@Author     ：ZAY
@Time       ：2023/6/5 16:18
@Annotation : " "
"""
from torch.utils.data import Dataset

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