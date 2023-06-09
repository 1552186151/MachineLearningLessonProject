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
import sklearn
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score,auc,roc_curve,precision_recall_curve,f1_score, precision_score, recall_score
from Exp.Exp3.Plot import plotSepalShow,plotPetalShow,plotSPShow

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将标签由文字映射为数字
def Iris_label(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]

def Iris_Sepal_Cla(data_x, label_y):
    data_x = data_x[:, 0:2]
    train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(data_x, label_y,
                                                                                              random_state = 1,
                                                                                              train_size = 0.8,
                                                                                              test_size = 0.2)
    # 训练 SVM 分类器 https://blog.csdn.net/TeFuirnever/article/details/99646257
    classifier = svm.SVC(C = 0.5, kernel = 'rbf', gamma = 10, decision_function_shape = 'ovr')  # rbf
    classifier.fit(train_data, train_label.ravel())

    train_label_pre = classifier.predict(train_data)
    test_label_pre = classifier.predict(test_data)
    print('花萼训练集acc:', accuracy_score(train_label, train_label_pre))
    print('花萼测试集acc:', accuracy_score(test_label, test_label_pre))

    # 查看内部决策函数（返回的是样本到超平面的距离）
    train_decision_function = classifier.decision_function(train_data)
    predict_result = classifier.predict(train_data)

    # print('train_decision_function:', train_decision_function)
    # print('predict_result:', predict_result)

    plotSepalShow(test_data, test_label, data_x, label_y, classifier)

def Iris_Petal_Cla(data_x, label_y):
    # 基于SVM鸢尾花瓣长宽度二特征分类
    data_x = data_x[:, 2:4]

    train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(data_x, label_y,
                                                                                              random_state = 1,
                                                                                              train_size = 0.8,
                                                                                              test_size = 0.2)
    classifier = svm.SVC(C = 0.5, kernel = 'rbf', gamma = 10, decision_function_shape = 'ovr')  # rbf
    classifier.fit(train_data, train_label.ravel())

    train_label_pre = classifier.predict(train_data)
    test_label_pre = classifier.predict(test_data)
    print('花瓣训练集acc:', accuracy_score(train_label, train_label_pre))
    print('花瓣测试集acc:', accuracy_score(test_label, test_label_pre))

    plotPetalShow(test_data, test_label, data_x, label_y, classifier)

def Iris_Sepal_Petal_Cla(data_x, label_y):
    data_x = np.stack((data_x[:, 0], data_x[:, 2], data_x[:, 3]), axis=1)

    train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(data_x, label_y,
                                                                                              random_state = 1,
                                                                                              train_size = 0.8,
                                                                                              test_size = 0.2)
    classifier = svm.SVC(C = 0.5, kernel = 'linear', gamma = 10, decision_function_shape = 'ovr')  # rbf
    classifier.fit(train_data, train_label.ravel())

    train_label_pre = classifier.predict(train_data)
    test_label_pre = classifier.predict(test_data)
    print('花萼和花瓣训练集acc:', accuracy_score(train_label, train_label_pre))
    print('花萼和花瓣测试集acc:', accuracy_score(test_label, test_label_pre))

    plotSPShow(classifier, data_x, label_y)


if __name__ == "__main__":

    txt_path = './/Result//SVM.txt'

    data = np.loadtxt("./Data/iris.data", dtype = float, delimiter = ',', converters = {4: Iris_label})
    # 基于SVM鸢尾花萼长宽度二特征分类
    data_x, label_y = np.split(data, indices_or_sections = (4,), axis = 1)  # x为数据，y为标签

    Iris_Sepal_Cla(data_x, label_y)

    Iris_Petal_Cla(data_x, label_y)

    Iris_Sepal_Petal_Cla(data_x, label_y)

