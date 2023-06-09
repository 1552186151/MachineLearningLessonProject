# -*- coding: utf-8 -*- #
"""
@Project    ：MachineLearningLesson
@File       ：LinearRegression_women.py
@Author     ：ZAY
@Time       ：2023/3/16 10:18
@Annotation : " "
"""

# packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv(".//Data//women.csv", index_col = 0)
X = data["height"]
# 添加截距项
X = sm.add_constant(X)
y = data["weight"]
# 数据描述性分析
data.describe()
# 绘制散点图
plt.scatter(data["height"], data["weight"], color = 'magenta')
# 添加标题
plt.xlabel('height')
plt.ylabel('weight')
plt.title('women-height-weight_scatter')
plt.show()
# 最小二成模型
model = sm.OLS(y, X)
# 训练模型
result = model.fit()
# 输出训练结果
print(result.summary())
# 模型预测
y_pre = result.predict()
print(y_pre)
# 结果可视化
plt.rcParams['font.family'] = "simHei"  # 汉字显示
plt.plot(data["height"], data["weight"], "o", color = 'magenta')
plt.plot(data["height"], y_pre)
plt.title('women-height-weight_scatter')
plt.show()
plt.savefig('.//Result//women-height-weight_LR_scatter.png')
