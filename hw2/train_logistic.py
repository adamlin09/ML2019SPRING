import csv, sys
import numpy as np
import math

x = []
y = []
test_x = []

X_train = open('./ml2019spring-hw2/X_train')
file_x = csv.reader(X_train)
for row in file_x :
    if row[0] == 'age' :
        continue
    x.append([])
    j = 0
    for i in row :
        x[-1].append(float(i))
        if j == 0 or j == 1 or j == 3 or j == 5 :
            x[-1].append(float(i)**2)
        j += 1
X_train.close()

Y_train = open('./ml2019spring-hw2/Y_train')
file_y = csv.reader(Y_train)
for row in file_y :
    if row[0] == 'label' :
        continue
    y.append(float(row[0]))
Y_train.close()

X_test = open('./ml2019spring-hw2/X_test')
testfile = csv.reader(X_test)
for row in testfile :
    if row[0] == 'age' :
        continue
    test_x.append([])
    j = 0
    for i in row :
        test_x[-1].append(float(i))
        if j == 0 or j == 1 or j == 3 or j == 5 :
            test_x[-1].append(float(i)**2)
        j += 1
X_test.close()

x = np.array(x)
y = np.array(y)
test_x = np.array(test_x)
x_all = np.concatenate((x, test_x))
#normalization
#mean = np.mean(x, axis=0)
#std = np.std(train_x, axis=0)
max = np.max(x_all, axis = 0)
min = np.min(x_all, axis = 0)
for i in range(x.shape[0]) :
    for j in range(x.shape[1]) :
        if max[j] != min[j] :
            x[i][j] = (x[i][j] - min[j]) / (max[j] - min[j])
np.save('max.npy', max)
np.save('min.npy', min)

r = int(0.8 * x.shape[0])
train_x = np.array(x[:r])
train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x), axis=1)
train_y = np.array(y[:r])
valid_x = np.array(x[r:])
valid_x = np.concatenate((np.ones((valid_x.shape[0],1)),valid_x), axis=1)
valid_y = np.array(y[r:])

w = np.ones(train_x.shape[1])
lr = 1
iteration = 100000
prev_grad = 0
s = 0.01
for i in range(iteration) :
    z = np.dot(train_x, w)
    f = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
    #L = -np.dot(np.transpose(np.log(f)), train_y) - np.dot(np.transpose(np.log(f)), (1 - train_y))
    grad_L = - np.dot(np.transpose(train_x), (train_y - f)) + 2 * s * np.sum(w)
    prev_grad += (grad_L ** 2)
    ada = np.sqrt(prev_grad)
    w -= lr * grad_L / (ada + 0.0005)
    if (i + 1) % 1000 == 0 :
        predict = 1 / (1 + np.exp(-(np.dot(valid_x, w))))
        for j in predict :
            if j > 0.5 :
                j = 1
            else :
                j = 0
        accuracy = 1 - np.sum(np.abs(predict - valid_y)) / len(valid_y)
        print(accuracy)
np.save('model_logistic.npy', w)