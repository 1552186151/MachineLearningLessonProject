# -*- coding: utf-8 -*- #
"""
@Project    ：MachineLearningLesson
@File       ：plot.py 
@Author     ：ZAY
@Time       ：2023/6/5 21:41
@Annotation : " "
"""

# 确定坐标轴范围
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plotSepalShow(test_data, test_label, data_x, label_y, classifier):
    x1_min, x1_max = data_x[:, 0].min(), data_x[:, 0].max()  # 第0维特征的范围
    x2_min, x2_max = data_x[:, 1].min(), data_x[:, 1].max()  # 第1维特征的范围
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
    # print(x1.shape) # (200, 200)
    # print(x1.flat) # flat属性可以使得像遍历以一维数组的方法来遍历多维数组
    grid_test = np.stack((x1.flat, x2.flat), axis = 1)  # 测试点
    # print(grid_test.shape) # (40000, 2)
    # 指定默认字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']

    # 设置颜色
    cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])

    grid_hat = classifier.predict(grid_test)  # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

    plt.pcolormesh(x1, x2, grid_hat, cmap = cm_light)  # 预测值的显示
    plt.scatter(data_x[:, 0], data_x[:, 1], c = label_y[:, 0], s = 30, cmap = cm_dark)  # 样本
    plt.scatter(test_data[:, 0], test_data[:, 1], c = test_label[:, 0], s = 30, edgecolors = 'k', zorder = 2,
                cmap = cm_dark)  # 圈中测试集样本点
    plt.xlabel('花萼长度', fontsize = 13)
    plt.ylabel('花萼宽度', fontsize = 13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('基于SVM鸢尾花萼长宽度二特征分类')
    plt.savefig('./Result/iris-sepal-cal.png')
    plt.show()

def plotPetalShow(test_data, test_label, data_x, label_y, classifier):
    x1_min, x1_max = data_x[:, 0].min(), data_x[:, 0].max()  # 第2维特征的范围
    x2_min, x2_max = data_x[:, 1].min(), data_x[:, 1].max()  # 第3维特征的范围
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
    # print(x1.shape) # (200, 200)
    # print(x1.flat) # flat属性可以使得像遍历以一维数组的方法来遍历多维数组
    grid_test = np.stack((x1.flat, x2.flat), axis = 1)  # 测试点
    # print(grid_test.shape) # (40000, 2)
    # 指定默认字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']

    # 设置颜色
    cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])

    grid_hat = classifier.predict(grid_test)  # 预测分类值
    grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

    plt.pcolormesh(x1, x2, grid_hat, cmap = cm_light)  # 预测值的显示
    plt.scatter(data_x[:, 0], data_x[:, 1], c = label_y[:, 0], s = 30, cmap = cm_dark)  # 样本
    plt.scatter(test_data[:, 0], test_data[:, 1], c = test_label[:, 0], s = 30, edgecolors = 'k', zorder = 2,
                cmap = cm_dark)  # 圈中测试集样本点
    plt.xlabel('花瓣长度', fontsize = 13)
    plt.ylabel('花瓣宽度', fontsize = 13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('基于SVM鸢尾花瓣长宽度二特征分类')
    plt.savefig('./Result/iris-petal-cla.png')
    plt.show()

def plotSPShow(clf, x, y):
    iris_feature = 'sepal length', 'sepal width', 'petal lenght', 'petal width'
    # 开始画图
    x0_min, x0_max = x[:, 0].min(), x[:, 0].max()
    x1_min, x1_max = x[:, 1].min(), x[:, 1].max()  # 第0列的范围
    x2_min, x2_max = x[:, 2].min(), x[:, 2].max()  # 第1列的范围
    x0, x1, x2 = np.mgrid[x0_min:x0_max:50j, x1_min:x1_max:50j, x2_min:x2_max:50j]  # 生成网格采样点,3D
    grid_test = np.stack((x0.flat, x1.flat, x2.flat), axis=1)  # stack():沿着新的轴加入一系列数组， flat的作用是将数组分解成可连续访问的元素,目的就是把他拉直后合并，并且不改变数组
    print('grid_test:\n', grid_test)

    grid_hat = clf.predict(grid_test)  # 预测分类值 得到【0,0.。。。2,2,2】
    print('grid_hat:\n', grid_hat)
    grid_hat = grid_hat.reshape(x1.shape)  # reshape grid_hat和x1形状一致
    # 若3*3矩阵e，则e.shape()为3*3,表示3行3列

    cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # s：marker标记的大小
    # c: 颜色  可为单个，可为序列
    # depthshade: 是否为散点标记着色以呈现深度外观。对 scatter() 的每次调用都将独立执行其深度着色。
    # marker：样式
    # alpha为点的透明度，在0~1之间

    ax.scatter(xs=x1, ys=x2, zs=x0, zdir='z', s=10, c=grid_hat, depthshade=True, cmap=cm_light,alpha=0.01)
    ax.scatter(xs=x[:,1], ys=x[:,2], zs=x[:,0], zdir='z', s=30, c=np.squeeze(y), depthshade=True, cmap=cm_dark, marker="^")
    plt.title('基于SVM鸢尾花萼长度和花瓣长宽度三特征分类')
    plt.savefig('./Result/iris-sepal-petal-cla.png')
    plt.show()
