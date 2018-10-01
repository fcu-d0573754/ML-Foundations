
# coding: utf-8

# """
# 1.初始化w0
# {
#     2.找到错误点
#     3.w(t+1)=w(t)+y*x
#     4.比较w(t+1)与w(t)对数据的错误率，保留好的
# }迭代次数=50
# """

# In[1]:

train_data_path = 'https://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw1/hw1_18_train.dat'
test_data_path = 'https://www.csie.ntu.edu.tw/~htlin/course/ml15fall/hw1/hw1_18_test.dat'


# In[2]:

import numpy as np
import pandas as pd

train_dataset = pd.read_table(train_data_path, sep='\s+',header=None)
train_X = train_dataset.iloc[:,:4].values
train_Y = train_dataset.iloc[:,-1].values
train_X

test_dataset = pd.read_table(test_data_path, sep='\s+', header=None)
test_X = test_dataset.iloc[:,:4].values
test_Y = test_dataset.iloc[:,-1].values
w = np.zeros((1,5))

train_x0 = np.ones((train_X.shape[0],1))
train_X = np.concatenate((train_X,train_x0),axis=1)
test_x0 = np.ones((train_X.shape[0],1))
test_X = np.concatenate((test_X,train_x0),axis=1)

best_w = w


def sign(x):
    if x > 0:
        return 1
    else:
        return -1


train_sample = train_Y.shape[0]

error_num = train_sample
Error = error_num

Error_sum = 0

for n in range(2000):
    permutation = np.permutation(train_sample)
    train_X = train_X[permutation,:]
    train_Y = np.array(train_Y[permutation])
    
    
    #開始進行迭代
    Error = train_sample
    for i in range(50):
        error_num = 0
        w_save = w
        for j in range(train_sample):
            if sign(np.dot(w,train_X[j].T)) != train_Y[j]:
                w = w + train_Y[j] * train_X[j]
                for k in range(train_sample):
                    if sign(np.dot(w,train_X[k].T)) != train_Y[k]:
                        error_num += 1
                break
        if error_num > Error:
            w = w_save
        
    Error_sum += Error

print(Error)
