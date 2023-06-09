# -*- coding: utf-8 -*- #
"""
@Project    ：MachineLearningLesson
@File       ：PolynomialRegression.py 
@Author     ：ZAY
@Time       ：2023/3/16 10:43
@Annotation : " "
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv(".//Data//women.csv", index_col = 0)
X = data["height"]
y = data["weight"]
# 构造三阶多项式
X = np.column_stack((X, np.power(X, 2), np.power(X, 3)))
# 添加截距项
X = sm.add_constant(X)

model = sm.OLS(y, X)
result = model.fit()
print(result.summary())
y_pre = result.predict()
print(y_pre)
# 结果可视化
plt.rcParams['font.family'] = "simHei"  # 汉字显示
plt.plot(data["height"], data["weight"], "o", color = 'magenta')
plt.plot(data["height"], y_pre)
plt.title('women-height-weight_scatter')
plt.show()
plt.savefig('.//Result//women-height-weight_PR_scatter.png')
