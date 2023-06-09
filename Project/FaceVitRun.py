# -*- coding: utf-8 -*- #
"""
@Project    ：MachineLearningLesson
@File       ：FaceVitRun.py
@Author     ：ZAY
@Time       ：2023/5/30 15:44
@Annotation : "运行Transformer模型 "
"""

import os
import glob
import torch
import torch.nn as nn
import datetime
import numpy as np
from timm.models.layers import to_2tuple
from torchvision import transforms
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from Project.Plot import plotloss,plotShow,plotROC
from Project.Net.VitNet import ViT
from DataLoad import  Mydataset, Mydatasetpro
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score,auc,roc_curve,precision_recall_curve,f1_score, precision_score, recall_score
from matplotlib import pyplot as plt


LR = 0.0001 # 0.0001
EPOCH = 100
BATCH_SIZE = 10
Test_Batch_Size = 6

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def modeltrian(image_size, ncls, psize, depth, heads, dim, mlp_dim, path, data_train, data_test):

    global train_time

    # data_train, data_test = TableDataLoad(tp, test_ratio, start, end, seed=80)
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)

    train_result_path = 'Result/Train/transfomertable.csv'
    test_result_path = 'Result/Test/transfomertable.csv'


    store_path = path

    model = ViT(
        num_classes = ncls,
        image_size = to_2tuple(image_size),  # image size is a tuple of (height, width)
        patch_size = to_2tuple(psize),    # patch size is a tuple of (height, width)
        dim = dim, #1024 self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        depth = depth, #encoder和decoder的深度
        heads = heads, #注意力机制的数量
        mlp_dim = mlp_dim, #2048 encoder注意力机制后接多层全连接层
        dropout = 0.1,
        emb_dropout = 0.1
        ).to(device)

    # summary(model, (1, 2000, 1), batch_size=1, device="cuda")

    criterion = nn.CrossEntropyLoss().to(device)  # 损失函数为焦损函数，多用于类别不平衡的多分类问题

    optimizer = optim.Adam(model.parameters(), lr=LR)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）

    avg_train_loss = []
    avg_test_loss = []
    train_sum = 0
    print("This is VitRun")
    print("Start Training!")  # 定义遍历数据集的次数
    with open(train_result_path, "w") as f1:
        with open(test_result_path, "w") as f2:
            f1.write("{},{},{}".format(("epoch"), ("loss"), ("acc")))  # 写入数据
            f1.write('\n')
            f2.write("{},{},{}".format(("epoch"), ("loss"), ("acc")))  # 写入数据
            f2.write('\n')
            startTime = datetime.datetime.now()
            for epoch in range(EPOCH):
                train_acc = []
                train_losses = []
                for i, data in enumerate(train_loader):  # gives batch data, normalize x when iterate train_loader
                    model.train()
                    inputs, labels = data  # 输入和标签都等于data
                    # print("inputs",inputs)
                    # print("inputs.shape", inputs.shape)
                    # print("labels",labels)
                    # print("labels.shape", labels.shape)
                    inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
                    labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
                    output = model(inputs)  # cnn output
                    # print(output.shape)  # torch.Size([10, 5])
                    # print(labels.shape)  # torch.Size([10])
                    print(output.data)
                    print(labels)
                    loss = criterion(output, labels)  # cross entropy loss
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients
                    _, predicted = torch.max(output.data,1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度
                    y_predicted = predicted.cpu().numpy()
                    label = labels.cpu().numpy()
                    print("Train: y_predicted",y_predicted,"label",label)
                    acc = accuracy_score(label, y_predicted)
                    # print("trian:epoch = {:} Loss = {:.4f}  Acc= {:.4f}".format((epoch + 1), (loss.item()),(acc)))  # 训练次数，总损失，精确度
                    # f1.write("{:},{:.4f},{:.4f}".format((epoch + 1), (loss.item()), (acc)))  # 写入数据
                    # f1.write('\n')
                    # f1.flush()
                    train_losses.append(loss.item())
                    train_acc.append(acc)
                train_sum = epoch + 1
                avg_train_loss.append(np.mean(train_losses))
                avg_acc = np.mean(train_acc)
                print('Epoch:{},Train Acc:{:.4f}'.format((epoch + 1),avg_acc))
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
                        _, predicted = torch.max(outputs.data,1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 ，取得分最高的那个类 (outputs.data的索引号)
                        y_predicted = predicted.cpu().numpy()
                        label = labels.cpu().numpy()
                        acc = accuracy_score(label, y_predicted)
                        test_loss.append(loss.item())
                        test_acc.append(acc)
                        # print("test:epoch = {:}   Acc= {:.4f}".format((epoch + 1) , (acc)))
                        # f2.write("{},{:.4f},{:.4f}".format((epoch + 1), (loss.item()), (acc)))  # 写入数据
                        # f2.write('\n')
                        # f2.flush()

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
    plotloss(x = range(train_sum), y1 = avg_train_loss, picd_path = ".//Result//Train//loss" + '.png',
             label = "Train_Loss_list")
    plotloss(x = range(train_sum), y1 = avg_test_loss, picd_path = ".//Result//Test//loss" + '.png',
             label = "Test_Loss_list")
    plotloss(x = range(train_sum), y1 = avg_train_loss, y2 = avg_test_loss,
             picd_path = ".//Result//loss" + '.png')
            # torch.save(model, './/model//transformer.pth')
            # 将每次测试结果实时写入acc.txt文件中


def modeltest(image_size, ncls, psize, depth, heads, dim, mlp_dim, path, data_test, txt_path):
    # _, data_test = DataLoad('tou', test_ratio, start, end)
    # data_train, data_test = TableDataLoad(tp, test_ratio, image_size, seed=80)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=Test_Batch_Size, shuffle=True)

    model = ViT(
        num_classes = ncls,
        image_size = to_2tuple(image_size),  # image size is a tuple of (height, width)
        patch_size = to_2tuple(psize),  # patch size is a tuple of (height, width)
        dim = dim,  # 1024 self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        depth = depth,  # encoder和decoder的深度
        heads = heads,  # 注意力机制的数量
        mlp_dim = mlp_dim,  # 2048 encoder注意力机制后接多层全连接层
        dropout = 0.1,
        emb_dropout = 0.1
    ).to(device)
    # store_path = './/model//all//transform'+'{}new.pt'.format(int(10-10*(test_ratio)))
    store_path = path
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
        acc_list.append(acc)
        label = judgeType(label)
        pre = judgeType(y_predicted)
        pre_list.append(pre)
        print("第{}张图片朝向为：{}".format(i+1, label))
        print("第{}张图片预测为：{}".format(i+1, pre))
    endTime = datetime.datetime.now()
    test_time = endTime - startTime
    plotShow(id_to_species,test_face_dataloader,pre_list)
    date = datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')  # 不能使用冒号
    log = open(txt_path, mode = "a+", encoding = "utf-8")
    print('DATE:{}, TEST:Acc= {:.4f}, train_time = {}, test_time = {}'.format(date,float(np.mean(acc_list)), 0, test_time),file = log)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"number of params: {n_parameters / 1e6} M")
    print(f"当前模型参数量: {n_parameters / 1e6} M",file = log)
    if hasattr(model, 'flops'):  # 判断某个类实例对象是否包含指定名称的属性或方法 返回boolean
        flops = model.flops()
        # print(f"number of MFLOPs: {flops / 1e6}")
        print(f"当前模型运算量: {flops / 1e6} MFLOPs",file = log)
    print("Acc= {:.4f}".format(np.mean(acc_list)))
    return np.mean(acc_list)

def calculate_fpr_tpr_tnr_f1score_accuracy(y_true, y_pred):
    '''
    y_true 和 y_pred均是ndarray类型。
    '''
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    Tp = 0
    Fp = 0
    Tn = 0
    Fn = 0
    for label, pred in zip(y_true, y_pred):
        if (label == 0) and (pred == 0):
            Tp = Tp + 1
        elif (label == 1) and (pred == 0):
            Fp = Fp + 1
        elif (label == 1) and (pred == 1):
            Tn = Tn + 1
        elif (label == 0) and (pred == 1):
            Fn = Fn + 1
        else:
            print('something weird with labels')
            return -1
            # sys.exit()
    # calculate precision, recall, accuracy, f1
    # it's possible for division by zero in some of these cases, so do a try/except
    try:
        precision = Tp / (Tp + Fp)
    except:
        precision = 0
    try:
        recall = Tp / (Tp + Fn)
    except:
        recall = 0
    try:
        accuracy = (Tn + Tp) / (Tn + Tp + Fn + Fp)
    except:
        accuracy = 0
    try:
        f1Score = 2 * precision * recall / (precision + recall)
    except:
        f1Score = 0
    try:
        fpr = Fp / (Fp + Tn)
    except:
        fpr = 0
    try:
        tpr = Tp / (Tp + Fn)
    except:
        tpr = 0
    try:
        tnr = Tn / (Tn + Fp)
    except:
        tnr = 0
    return (fpr, tpr, tnr, f1Score, accuracy)

def judgeType(y_predicted):
    if y_predicted == 0:
        pre = "left"
    elif y_predicted == 1:
        pre = "lf"
    elif y_predicted == 2:
        pre = "front"
    elif y_predicted == 3:
        pre = "rf"
    else:
        pre = "right"
    return pre

class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        return str(acc)

    def plot(self):  # 绘制混淆矩阵
        matrix = self.matrix
        print("matrix: ",matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + self.summary() + ')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.savefig(".//Result//matrix.png")
        plt.show()

def model4AUCtest(image_size, ncls, psize, depth, heads, dim, mlp_dim, path, data_test, txt_path):
    #_, data_test = DataLoad(tp, test_ratio, start, end)
    # data_train, data_test = TableDataLoad(tp, test_ratio, start, end, seed=80)
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=Test_Batch_Size, shuffle=False)
    model = ViT(
        num_classes = ncls,
        image_size = to_2tuple(image_size),  # image size is a tuple of (height, width)
        patch_size = to_2tuple(psize),  # patch size is a tuple of (height, width)
        dim = dim,  # 1024 self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        depth = depth,  # encoder和decoder的深度
        heads = heads,  # 注意力机制的数量
        mlp_dim = mlp_dim,  # 2048 encoder注意力机制后接多层全连接层
        dropout = 0.1,
        emb_dropout = 0.1
        ).to(device)
    # store_path = './/model//all//transform'+'{}new.pt'.format(int(10-10*(test_ratio)))
    store_path = path
    model.load_state_dict(torch.load(store_path))
    labels = [0, 1, 2, 3, 4]
    acc_list = []
    pre_list = []
    # tomato_DICT = {'0': 'Bacterial_spot', '1': 'Early_blight', '2': 'healthy', '3': 'Late_blight', '4': 'Leaf_Mold'}
    # label = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=5, labels=labels)
    startTime = datetime.datetime.now()  # 计算训练时间
    log = open(txt_path, mode = "a+", encoding = "utf-8")
    for i, data in enumerate(test_loader):
        model.eval()  # 不训练
        inputs, labels = data  # 输入和标签都等于data
        inputs = Variable(inputs).type(torch.FloatTensor).to(device)  # batch x
        labels = Variable(labels).type(torch.LongTensor).to(device)  # batch y
        outputs = model(inputs)  # 输出等于进入网络后的输入
        y_proba = outputs.data.cpu().numpy()
        _, predicted = torch.max(outputs.data,1)  # _ , predicted这样的赋值语句，表示忽略第一个返回值，把它赋值给 _， 就是舍弃它的意思，预测值＝output的第一个维度 ，取得分最高的那个类 (outputs.data的索引号)
        y_predicted = predicted.cpu().numpy()
        label = labels.cpu().numpy()

        y_one_hot = label_binarize(label, classes=[0, 1, 2, 3, 4])
        # ACC
        acc = accuracy_score(label, y_predicted)
        precis = precision_score(label, y_predicted, average='weighted') # precision：精准度（true positive/true positive+false positive）在所有分类为x类的样本中确实为x类的比例；
        reca = recall_score(label, y_predicted, average='weighted') # recall：召回率（true positive/true positive+false negative）在所有标签为x类的样本中分类为x类的比例；

        acc_list.append(acc)
        labels_name = [0, 1, 2, 3, 4]
        arry = metrics.confusion_matrix(y_true=label, y_pred=y_predicted, labels=labels_name)  # 生成混淆矩阵

        confusion.update(y_predicted, label)

        # FPR,TPR
        false_positive_rate, true_positive_rate, _ = roc_curve(y_one_hot.ravel(), y_proba.ravel())
        # new_tpr
        mean_fpr = np.linspace(0, 1, 100) # 生成等间距数组 np.linspace(start, stop, num, endpoint, retstep, dtype)
        new_true_positive_rate = np.interp(mean_fpr, false_positive_rate, true_positive_rate) # 一维线性插值 https://blog.csdn.net/hfutdog/article/details/87386901
        # AUC
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Recall、Precision
        precision, recall, _ = precision_recall_curve(y_one_hot.ravel(), y_proba.ravel())
        # new_recall
        mean_recall = np.linspace(0, 1, 100)
        new_precision = np.interp(mean_recall, precision, recall)
        # new_precision = np.interp(mean_recall, recall, precision)
        # F1 宏平均、微平均、权重评价即Macro-average、Micro-Average，Weighted-Average https://blog.csdn.net/niutingbaby/article/details/106943394
        F1 = f1_score(label, y_predicted, average='weighted') # f1-score：F1值，（2/F1 = 1/P + 1/R，F1 = 2PR/(P+R)）精准度和召回率的调和平均值，精准度和召回率都高的时候F1值也高
        # F2 = f1_score(y_test,y_pred,average='macro')
        # F3 = f1_score(y_test,y_pred,average='micro')
        # (fpr, tpr, tnr, f1Score, accuracy) = calculate_fpr_tpr_tnr_f1score_accuracy(np.array(label), np.array(y_predicted))

        plotROC(false_positive_rate, true_positive_rate, roc_auc)
        for i in range(len(label)):
            lab = judgeType(label[i])
            pre = judgeType(y_predicted[i])
            print("第{}张图片朝向为：{}".format(i + 1, lab))
            print("第{}张图片预测为：{}".format(i + 1, pre))
            pre_list.append(pre)
            print("第{}张图片朝向为：{}, 第{}张图片预测为：{}".format(i + 1, lab, i + 1, pre), file = log)

        print("acc:{}, precis:{}, recall:{}, F1:{}, auc:{}".format(acc, precis, reca, F1, roc_auc))
        # print("false_positive_rate:{}, true_positive_rate:{}, precision:{}, recall:{}".format(false_positive_rate, true_positive_rate, precision, recall))
        print("acc:{}, precis:{}, recall:{}, F1:{}, auc:{}".format(acc, precis, reca, F1, roc_auc), file = log)
    endTime = datetime.datetime.now()
    test_time = endTime - startTime
    date = datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')  # 不能使用冒号
    plotShow(id_to_species,test_face_dataloader,label_batch = pre_list)
    print('DATE:{}, TEST:Avg_acc= {:.4f}, train_time = {}, test_time = {}'.format(date, float(np.mean(acc_list)), train_time, test_time), file = log)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"number of params: {n_parameters / 1e6} M")
    print(f"当前模型参数量: {n_parameters / 1e6} M", file = log)
    if hasattr(model, 'flops'):  # 判断某个类实例对象是否包含指定名称的属性或方法 返回boolean
        flops = model.flops()
        # print(f"number of MFLOPs: {flops / 1e6}")
        print(f"当前模型运算量: {flops / 1e6} MFLOPs", file = log)
    print("Avg_acc= {:.4f}".format(np.mean(acc_list)))
        # # runing_time
        # run_time = end - start
    confusion.plot()
    confusion.summary()

    return acc, precis, reca, F1, roc_auc #new_true_positive_rate, new_precision


if __name__ == "__main__":

    global id_to_species

    store_path = 'model/transformer.pt'
    txt_path = 'Result/Vit.txt'

    # 使用glob方法来获取数据图片的所有路径
    all_imgs_path = glob.glob(r"Project/Data/*/*.bmp")  # 数据文件夹路径

    # for var in all_imgs_path:
    #     print(var)

    # 利用自定义类Mydataset创建对象face_dataset
    face_dataset = Mydataset(all_imgs_path)
    print("文件夹中图片总个数:",len(face_dataset))  # 返回文件夹中图片总个数
    # face_dataloader = torch.utils.data.DataLoader(face_dataset, batch_size = 8)  # 每次迭代时返回8个数据
    # 为每张图片制作对应标签
    species = ['left', 'lf', 'front', 'rf', 'right']
    species_to_id = dict((c, i) for i, c in enumerate(species))
    id_to_species = dict((v, k) for k, v in species_to_id.items())
    print("id_to_species",id_to_species)

    # 对所有图片路径进行迭代
    all_labels = []
    for img in all_imgs_path:
        # 区分出每个img，应该属于什么类别
        for i, c in enumerate(species):
            if c in img:
                all_labels.append(i)
    print("all_labels",all_labels)

    # 对数据进行转换处理
    transform = transforms.Compose([
        transforms.Resize((420, 420)),  # 做的第一步转换
        transforms.ToTensor()  # 第二步转换，作用：第一转换成Tensor，第二将图片取值范围转换成0-1之间，第三会将channel置前
    ])

    face_dataset = Mydatasetpro(all_imgs_path, all_labels, transform)
    face_dataloader = DataLoader(
        face_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    imgs_batch, labels_batch = next(iter(face_dataloader))
    print("imgs_batch.shape",imgs_batch.shape) # torch.Size([4, 3, 420, 420])


    # plt.figure(figsize = (12, 8))
    # for i, (img, label) in enumerate(zip(imgs_batch[:6], labels_batch[:6])):
    #     img = img.permute(1, 2, 0).numpy() # (H,W,C)
    #     plt.subplot(2, 3, i + 1) # subplot(numRows, numCols, plotNum) numRows 行 numCols 列
    #     plt.title(id_to_species.get(label.item()))
    #     plt.imshow(img)
    # plt.show()  # 展示图片

    # 划分数据集和测试集
    index = np.random.permutation(len(all_imgs_path))
    print("index",index)
    # 打乱顺序
    all_imgs_path = np.array(all_imgs_path)[index]
    all_labels = np.array(all_labels)[index]

    for i in range(len(all_imgs_path)):
        print("第{}张图片存储路径为：{}, 朝向为：{}, 标签为：{}".format(i + 1, all_imgs_path[i],judgeType(all_labels[i]),all_labels[i]))
    # 80%做训练集
    # c = int(len(all_imgs_path) * 0.8)
    # print("训练集和验证集数量:", c)
    c = 40
    v = 4
    t = 6

    c_imgs = all_imgs_path[:c]
    c_labels = all_labels[:c]
    v_imgs = all_imgs_path[c:c+v]
    v_labels = all_labels[c:c+v]
    t_imgs = all_imgs_path[c+v:]
    t_labels = all_labels[c+v:]

    test_face_dataset = Mydatasetpro(t_imgs, t_labels, transform)
    global test_face_dataloader
    test_face_dataloader = DataLoader(
        test_face_dataset,
        batch_size = t,
        shuffle = False # 在每个epoch开始的时候是否进行数据的重新排序，默认false
    )

    # train_imgs = all_imgs_path[:c]
    # train_labels = all_labels[:c]
    # test_imgs = all_imgs_path[c:]
    # test_labels = all_labels[c:]

    # print(test_imgs)
    # print(test_imgs.shape)
    # print(test_labels)
    # print(test_labels.shape)

    cal_ds = Mydatasetpro(c_imgs, c_labels, transform)  # TrainSet TensorData
    val_ds = Mydatasetpro(v_imgs, v_labels, transform)  # TestSet TensorData
    test_ds = Mydatasetpro(t_imgs, t_labels, transform)

    # print(next(iter(face_dataloader)))
    # 改进 psize 30-60 减少必要的epoch heads 12-6 减少参数和必要的epoch
    modeltrian(image_size = 420, ncls=5, psize=60, depth=3, heads=12, dim = 2048, mlp_dim=1024, path=store_path, data_train = cal_ds, data_test = val_ds)  # depth=6, heads=10, 12, 14

    # modeltest(image_size = 420, ncls = 5, psize = 60, depth=3, heads=6, dim = 2048, mlp_dim=1024, path=store_path, data_test = test_ds, txt_path = txt_path)

    acc, precis, reca, F1, roc_auc = model4AUCtest(image_size = 420, ncls=5, psize=60, depth=3, heads=12, dim = 2048, mlp_dim=1024, path=store_path, data_test = test_ds, txt_path = txt_path)  # depth=6, heads=10, 12, 14
    # print("acc:{}, precis:{}, recall:{}, F1:{}, auc:{}".format(acc, precis, reca, F1, roc_auc))