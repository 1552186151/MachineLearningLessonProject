# -*- coding: utf-8 -*- #
"""
@Project    ：MachineLearningLesson
@File       ：MultipleRegression_adver.py 
@Author     ：ZAY
@Time       ：2023/3/16 11:18
@Annotation : " "
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # 引用线性回归
from sklearn.model_selection import train_test_split # 对数据进行测试集与训练集的界分，有助于我们评价模型
from sklearn.metrics import mean_squared_error # 引用计算均平方差

data = pd.read_csv(".//Data//Advertising.csv", index_col = 0)
x = data[['TV','radio','newspaper']]
y = data["sales"]
plt.scatter(data["TV"],data["sales"],color='magenta')
plt.xlabel('TV')
plt.ylabel('sales')
plt.title('TV-sales_scatter')
plt.savefig('./Result/TV-sales_scatter.png')
plt.show()
plt.scatter(data["radio"],data["sales"],color='magenta')
plt.xlabel('radio')
plt.ylabel('sales')
plt.title('radio-sales_scatter')
plt.savefig('./Result/radio-sales_scatter.png')
plt.show()
plt.scatter(data["newspaper"],data["sales"],color='magenta')
plt.xlabel('newspaper')
plt.ylabel('sales')
plt.title('newspaper-sales_scatter')
plt.savefig('./Result/newspaper-sales_scatter.png')
plt.show()
x_train,x_test,y_train,y_test = train_test_split(x,y)
print("训练样本数量：" + str(len(x_train)) + " 测试样本数量：" + str(len(x_test)))
# 初始化模型
model = LinearRegression()
# 训练模型
model.fit(x_train,y_train)
print("斜率：" + str(model.coef_) + "截距：" + str(model.intercept_))
# 预测并计算均方差
print("MSE：",mean_squared_error(model.predict(x_test),y_test))
print("多元线性回归公式：y=" + str(round(model.coef_[0],3)) + "TV+" + str(round(model.coef_[1],3)) + "radio" + str(round(model.coef_[2],3)) + "newspaper+" + str(round(model.intercept_,3)))
print()
# 进一步优化 去除newspaper影响较小的自变量
x = data[['TV','radio']]
y = data["sales"]
x_train,x_test,y_train,y_test = train_test_split(x,y)
print("训练样本数量：" + str(len(x_train)) + " 测试样本数量：" + str(len(x_test)))
# 初始化模型
model = LinearRegression()
# 训练模型
model.fit(x_train,y_train)
print("斜率：" + str(model.coef_) + "截距：" + str(model.intercept_))
# 预测并计算均方差
print("MSE：",mean_squared_error(model.predict(x_test),y_test))
print("多元线性回归公式：y=" + str(round(model.coef_[0],3)) + "TV+" + str(round(model.coef_[1],3)) + "radio+" + str(round(model.intercept_,3)))
