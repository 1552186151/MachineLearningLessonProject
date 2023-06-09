# -*- coding: utf-8 -*- #
"""
@Project    ：MachineLearningLesson
@File       ：BPNet.py 
@Author     ：ZAY
@Time       ：2023/6/4 15:56
@Annotation : " "
"""

import torch
import torch.nn as nn


class BPNet(nn.Module):
    def __init__(self):
        super(BPNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(24, 96),
            nn.ReLU()
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(100, 50),
        #     nn.Sigmoid()
        # )

        self.predict = nn.Linear(96, 4)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.predict(self.fc(x)))
        return x


# net = BPNet()
# print(net)
