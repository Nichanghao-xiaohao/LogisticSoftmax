# -*- coding: utf-8 -*-
__author__ = 'Nichanghao'
__date__ = '2020/6/30'
__project__ = 'softmax'
__filename__ = 'softmax'

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model.coordinate_descent import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'SimHei'
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings(action='ignore',category=ConvergenceWarning)


#1、数据集在机器学习库中
#链接：https://archive.ics.uci.edu/ml/datasets/Credit+Approval

#2、读入数据
path = '../CreditApproval/crx.data'
names = ['A1','A2','A3','A4','A5','A6','A7','A8',
         'A9','A10','A11','A12','A13','A14','A15','A16']
df = pd.read_csv(path,header=None,names=names)
print(df.head(5))
## print出的数据类型不一致
# print(df.A4.value_counts())
# print(df.shape)
# print(len(df))
# print(df.describe().T)
#3、数据进行清洗
new_df = df.replace('?',np.nan)
new_df = new_df.dropna(axis=0,how='any')
# print(new_df.shape)
print('清洗过的数据')
# print(new_df.A14.value_counts())


# 手动实现哑编码
def parse(v,l):
    return [1 if i == v else 0 for i in l ]

def parseRecord(record):
    result = []
    a1 = record['A1']
    for i in parse(a1,('a','b')):
        result.append(i)

    result.append(float(record['A2']))
    result.append(float(record['A3']))

    a4 = record['A4']
    for i in parse(a4,('u','y','l','t')):
        result.append(i)

    a5 = record['A5']
    for i in parse(a5,('g','p','gg')):
        result.append(i)

    a6 = record['A6']
    for i in parse(a6, ('c', 'q', 'w','i','aa','ff','k','cc','m','x','d','e','j','r')):
        result.append(i)

    a7 = record['A7']
    for i in parse(a7, ('v', 'h', 'ff','bb','z','j','dd','n','o')):
        result.append(i)

    result.append(float(record['A8']))

    a9 = record['A9']
    for i in parse(a9, ('t', 'f')):
        result.append(i)

    a10 = record['A10']
    for i in parse(a10, ('t', 'f')):
        result.append(i)

    result.append(float(record['A11']))

    a12 = record['A12']
    for i in parse(a12, ('t', 'f')):
        result.append(i)

    a13 = record['A13']
    for i in parse(a13, ('g', 's','p')):
        result.append(i)

    result.append(float(record['A14']))
    result.append(float(record['A15']))

    a16 = record['A16']
    if a16 == '+':
        result.append(1)
    else:
        result.append(0)

    return result

new_names = ['A1_0','A1_1',
             'A2','A3',
             'A4_0','A4_1','A4_2','A4_3',
             'A5_0','A5_1','A5_2',
             'A6_0','A6_1','A6_2','A6_3','A6_4','A6_5','A6_6',
             'A6_7','A6_8','A6_9','A6_10','A6_11','A6_12','A6_13',
             'A7_0','A7_1','A7_2','A7_3','A7_4','A7_5','A7_6','A7_7','A7_8',
             'A8',
             'A9_0','A9_1',
             'A10_0','A10_1',
             'A11',
             'A12_0', 'A12_1',
             'A13_0', 'A13_1','A13_2',
             'A14',
             'A15',
             'A16'
             ]

datas = new_df.apply(lambda x:pd.Series(parseRecord(x),index=new_names),axis=1)
names = new_names
print(datas.head(5))
# print(names[0:-1])

# 数据分割
x = datas[names[0:-1]]
y = datas[names[-1]]
# print(x)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.9,random_state=0)

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

lr = LogisticRegressionCV(Cs=np.logspace(-4,1,50),penalty='l2',solver='lbfgs',fit_intercept= True,tol=0.01,multi_class='ovr')
lr.fit(x_train,y_train)

lr_r = lr.score(x_train,y_train)
lr_r_test = lr.score(x_test,y_test)

print('训练集的准确率R^2:%.5f' % lr_r)
print('测试集的准确率R^2:%.5f' % lr_r_test)
print('稀疏化特征比例:%.5f%%' % (np.mean(lr.coef_.ravel() == 0) * 100))
print('参数：',lr.coef_)
print('截距：',lr.intercept_)

y_predict = lr.predict(x_test)
print('预测所属类别：',y_predict)

print(lr.predict_proba(x_test))

x_len = range(len(x_test))
plt.figure(figsize = (14,7),facecolor='w')
plt.plot(x_len,y_test,'ro',markersize = 6,zorder = 3, label = u'真实值')
plt.plot(x_len,y_predict,'go',markersize = 10,zorder = 2,label = u'逻辑回归预测值(测试集),$R^2$=%.5f' % lr_r)
plt.legend(loc='center right')
plt.xlabel(u'数据编号ID',fontsize = 18 )
plt.ylabel(u'是否审批(0表示不通过，1表示通过)',fontsize = 18)
plt.title(u'Logistic回归算法实现',fontsize = 20)
plt.show()

