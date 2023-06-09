# -*- coding: utf-8 -*- #
"""
@Project    ：MachineLearningLesson
@File       ：main.py 
@Author     ：ZAY
@Time       ：2023/6/4 15:44
@Annotation : " "
"""

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score,auc,roc_curve,precision_recall_curve,f1_score, precision_score, recall_score
import torch.optim as optim
import datetime
import numpy as np
import scipy.io as scio
from Exp.Exp2.DataLoad import MyDataset
from Net.BPNet import BPNet

LR = 0.0001 # 0.0001
EPOCH = 300
BATCH_SIZE = 512
Test_Batch_Size = 512

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MapMinMaxApplier(object):
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def __call__(self, x):
        return x * self.slope + self.intercept

    def reverse(self, y):
        return (y - self.intercept) / self.slope

    def mapminmax(x, ymin = -1, ymax = +1):
        x = np.asanyarray(x)
        xmax = x.max(axis = -1)
        xmin = x.min(axis = -1)
        if (xmax == xmin).any():
            raise ValueError("some rows have no variation")
        slope = ((ymax - ymin) / (xmax - xmin))[:, np.newaxis]
        intercept = (-xmin * (ymax - ymin) / (xmax - xmin))[:, np.newaxis] + ymin
        ps = MapMinMaxApplier(slope, intercept)
        return ps(x), ps


if __name__ == "__main__":

    store_path = './/model//BPNet.pt'
    txt_path = './/Result//BPNet.txt'

    # 下载四类语音信号
    c1 = scio.loadmat('.//Data//data1.mat')
    c2 = scio.loadmat('.//Data//data2.mat')
    c3 = scio.loadmat('.//Data//data3.mat')
    c4 = scio.loadmat('.//Data//data4.mat')

    # print(type(c1)) # <class 'dict'>
    c1 = c1.get('c1')
    c2 = c2.get('c2')
    c3 = c3.get('c3')
    c4 = c4.get('c4')

    c1 = np.array(c1)
    c2 = np.array(c2)
    c3 = np.array(c3)
    c4 = np.array(c4)
    # print(c1.shape) # (500, 25)

    # scio.savemat(New_path, {'A': a, b}) # 以字典的形式保存

    # 四个特征信号矩阵合成一个矩阵
    data = np.vstack((c1[:, :], c2[:, :])) # 按垂直方向（行顺序）堆叠数组构成一个新的数组
    data = np.vstack((data[:, :], c3[:, :]))
    data = np.vstack((data[:, :], c4[:, :]))
    # print(data.shape) # (2000, 25)

    # 从1到2000间随机排序
    k = np.random.rand(2000)
    n = np.argsort(k) # 将k中的元素从小到大排列，提取其在排列前对应的index(索引)输出。
    print(n)

    # 输入输出数据
    input_data = data[:, 1:25]
    output_data = data[:, 0]
    # output_data = np.expand_dims(output_data,axis = 1)
    # print(output_data.shape) # (2000, 1)
    # print(output_data)

    # 把输出label从[1,2,3,4]变成[0,1,2,3]
    for i in range(2000):
        output_data[i] = output_data[i] - 1

    # 把输出从1维变成4维
    # output = np.zeros((2000, 4))
    # for i in range(2000):
    #     if output_data[i] == 1:
    #         output[i, :] = [1, 0, 0, 0]
    #     elif output_data[i] == 2:
    #         output[i, :] = [0, 1, 0, 0]
    #     elif output_data[i] == 3:
    #         output[i, :] = [0, 0, 1, 0]
    #     elif output_data[i] == 4:
    #         output[i, :] = [0, 0, 0, 1]

    output = output_data

    # 随机提取1500个样本为训练样本，500个样本为预测样本
    input_train = input_data[n[:1400], :]
    output_train = output[n[:1400]]
    input_val = input_data[n[1400:1500], :]
    output_val = output[n[1400:1500]]
    input_test = input_data[n[1500:], :]
    output_test = output[n[1500:]]

    # print(input_train.shape) # (1400, 24)
    # print(output_train.shape) # (1400, 1)
    # print(input_val.shape) # (100, 24)
    # print(output_val.shape) # (100, 1)
    # print(input_test.shape) # (500, 24)
    # print(output_test.shape) # (500, 1)

    # input_train = input_train[:, np.newaxis, :]
    # input_val = input_val[:, np.newaxis, :]
    # 输入数据归一化
    # inputn, inputps = MapMinMaxApplier.mapminmax(input_train)  # #默认为-1,1

    cal_ds = MyDataset(input_train, output_train)  # TrainSet TensorData
    val_ds = MyDataset(input_val, output_val)  # TestSet TensorData
    test_ds = MyDataset(input_test, output_test)

    # data_train, data_test = TableDataLoad(tp, test_ratio, start, end, seed=80)
    train_loader = torch.utils.data.DataLoader(cal_ds, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = torch.utils.data.DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle = True)

    train_result_path = './/Result//Train//BPNet.csv'
    test_result_path = './/Result//Test//BPNet.csv'

    model = BPNet().to(device)

    criterion = nn.CrossEntropyLoss().to(device)  # 损失函数为焦损函数，多用于类别不平衡的多分类问题

    optimizer = optim.Adam(model.parameters(), lr = LR)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    # early_stopping = EarlyStopping(patience = 30, delta = 1e-4, path = store_path, verbose = False)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = 0.5, verbose = 1, eps = 1e-07,
    #                                                        patience = 10)
    avg_train_loss = []
    avg_test_loss = []
    train_sum = 0

    # 训练模型
    startTime = datetime.datetime.now()
    for epoch in range(EPOCH):
        train_acc = []
        train_losses = []
        for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
            model.train()
            inputs, labels = data  # 输入和标签都等于data
            # print("inputs",inputs)
            # print("inputs.shape", inputs.shape) # torch.Size([128, 24])
            # print("labels",labels)
            # print("labels.shape", labels.shape)
            inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
            labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
            output = model(inputs)  # cnn output
            # print("output.shape", output.shape)  # torch.Size([128, 4])
            # print("labels.shape", labels.shape)  # torch.Size([128])
            # print("output",output)
            # print("labels",labels)
            loss = criterion(output, labels)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            print(output.data)
            _, predicted = torch.max(output.data,1)
            y_predicted = predicted.cpu().numpy()
            label = labels.cpu().numpy()
            print("Train: y_predicted", y_predicted, "label", label)
            acc = accuracy_score(label, y_predicted)
            train_losses.append(loss.item())
            train_acc.append(acc)
        train_sum = epoch + 1
        avg_train_loss.append(np.mean(train_losses))
        avg_acc = np.mean(train_acc)
        print('Epoch:{},Train Acc:{:.4f}'.format((epoch + 1), avg_acc))
        print('lr:{}, avg_train_loss:{}'.format((optimizer.param_groups[0]['lr']), avg_train_loss[epoch]))
        with torch.no_grad():  # 无梯度
            test_loss = []
            test_acc = []
            for i, data in enumerate(test_loader):
                model.eval()  # 不训练
                inputs, labels = data  # 输入和标签都等于data
                inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
                outputs = model(inputs)  # 输出等于进入网络后的输入
                loss = criterion(outputs, labels)  # cross entropy loss
                _, predicted = torch.max(outputs.data,1)
                y_predicted = predicted.cpu().numpy()
                label = labels.cpu().numpy()
                acc = accuracy_score(label, y_predicted)
                test_loss.append(loss.item())
                test_acc.append(acc)

            avg_test_loss.append(np.mean(test_loss))
            avg_acc = np.mean(test_acc)
            print('Epoch:{},Test Acc:{:.4f}'.format((epoch + 1), avg_acc))
            # scheduler.step(avgmse)  # 调整学习率
            # early_stopping(avgrmse, model)
            # if early_stopping.early_stop:
            #     print(f'Early stopping! Best validation loss: {early_stopping.get_best_score()}')
            #     break
    endTime = datetime.datetime.now()
    train_time = endTime - startTime
    torch.save(model.state_dict(), store_path)

    # 测试模型
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size = Test_Batch_Size, shuffle = True)

    model = BPNet().to(device)
    # store_path = './/model//all//transform'+'{}new.pt'.format(int(10-10*(test_ratio)))
    model.load_state_dict(torch.load(store_path))
    acc_list = []
    pre_list = []
    startTime = datetime.datetime.now()  # 计算训练时间
    for i, data in enumerate(test_loader):
        model.eval()  # 不训练
        inputs, labels = data  # 输入和标签都等于data
        inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
        labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
        outputs = model(inputs)  # 输出等于进入网络后的输入
        _, predicted = torch.max(outputs.data,1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 ，取得分最高的那个类 (outputs.data的索引号)
        y_predicted = predicted.cpu().numpy()
        label = labels.cpu().numpy()
        acc = accuracy_score(label, y_predicted)
        precis = precision_score(label, y_predicted,average = 'weighted')  # precision：精准度（true positive/true positive+false positive）在所有分类为x类的样本中确实为x类的比例；
        reca = recall_score(label, y_predicted,average = 'weighted')  # recall：召回率（true positive/true positive+false negative）在所有标签为x类的样本中分类为x类的比例；
        F1 = f1_score(label, y_predicted, average = 'weighted')
        acc_list.append(acc)
    endTime = datetime.datetime.now()
    test_time = endTime - startTime
    # plotShow(id_to_species,test_face_dataloader,pre_list)
    date = datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')  # 不能使用冒号
    log = open(txt_path, mode = "a+", encoding = "utf-8")
    print('DATE:{}, TEST:Acc:{:.4f}, precis:{:.4f}, recall:{:.4f}, F1:{:.4f}, train_time = {}, test_time = {}'.format(date, float(np.mean(acc_list)), precis, reca, F1, train_time,test_time))
    print('DATE:{}, TEST:Acc:{:.4f}, train_time = {}, test_time = {}'.format(date, float(np.mean(acc_list)), train_time,test_time), file = log)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"number of params: {n_parameters / 1e6} M")
    print(f"当前模型参数量: {n_parameters / 1e6} M", file = log)
    if hasattr(model, 'flops'):  # 判断某个类实例对象是否包含指定名称的属性或方法 返回boolean
        flops = model.flops()
        # print(f"number of MFLOPs: {flops / 1e6}")
        print(f"当前模型运算量: {flops / 1e6} MFLOPs", file = log)
    # print("Acc= {:.4f}".format(np.mean(acc_list)))



