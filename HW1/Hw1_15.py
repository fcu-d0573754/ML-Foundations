# -*- coding: utf-8 -*-
"""
{ 
1.寻找w(t)的下一个错误分类点（x,y）（即sign（w(t)’*x）！=y）； 
2.纠正错误：w(t+1) = w(t) + y*x； 
}until(每个样本都无错)

"""
import pandas as pd
import numpy as np


def sign(x):
    if x > 0:
        return 1
    else:
        return -1


path = 'https://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw1/hw1_15_train.dat'
dataset = pd.read_table(path, sep='\s+', header=None)

X = dataset.iloc[:, :4]
Y = dataset.iloc[:, -1]

w = np.zeros((1, 5))
x0 = np.ones((len(X), 1))
X = np.concatenate((X, x0), axis=1)

permutation = np.random.permutation(X.shape[0])
X = X[permutation,:]
Y = Y[permutation]
Y = np.array(Y)

sum_c = 0
list_count = []
for j in range(2000):
    permutation = np.random.permutation(X.shape[0])
    X = X[permutation,:]
    Y = Y[permutation]
    w = np.zeros((1, 5))
    w = w[0,:]
    error = 1
    count = 0
    while error == 1:
        error = 0
        i_random = np.random.permutation(X.shape[0])
        for i in i_random:
            y = np.dot(w, X[i, :].T)
            if sign(y) != Y[i]:
                #print('ok')
                w = w + Y[i] * X[i, :]
                error = 1
                count += 1
                break  
    #print(str(j) + "number of update: " + str(count))
    list_count.append(count)
    sum_c +=count
print(sum_c/2000)

import seaborn as sns
sns.set()
list_count = np.array(list_count)
ax = sns.distplot(list_count,bins = 30)



































