import csv, sys
import numpy as np
import math
from numpy.linalg import inv

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
#max = np.load('max.npy')
#min = np.load('min.npy')
mean = np.mean(x, axis=0)
std = np.std(x, axis=0)
for i in range(x.shape[0]) :
    for j in range(x.shape[1]) :
        if std[j] != 0 :
            x[i][j] = (x[i][j] - mean[j]) / std[j]
np.save('mean.npy', mean)
np.save('std.npy', std)

r = int(0.8 * x.shape[0])
train_x = np.array(x[:r])
train_y = np.array(y[:r])
valid_x = np.array(x[r:])
valid_y = np.array(y[r:])

over_50K = []
under_50K = []
index = 0
for i in train_y :
    if i == 1 :
        over_50K.append(train_x[index])
    else :
        under_50K.append(train_x[index])
    index += 1
over_50K = np.array(over_50K)
under_50K = np.array(under_50K)

#mean
over_50K_mean = np.mean(over_50K, axis=0)
under_50K_mean = np.mean(under_50K, axis=0)

#covarian
over_50K_cov = 0.0
under_50K_cov = 0.0
n1, n2 = float(over_50K.shape[0]), float(under_50K.shape[0])
for i in range(over_50K.shape[0]) :
    #over_50K_cov += np.dot((over_50K[i] - over_50K_mean), np.transpose(over_50K[i] - over_50K_mean))
    over_50K_cov += np.dot(np.transpose([over_50K[i] - over_50K_mean]), [(over_50K[i] - over_50K_mean)]) / n1
for i in range(under_50K.shape[0]) :
    under_50K_cov += np.dot(np.transpose([under_50K[i] - under_50K_mean]), [(under_50K[i] - under_50K_mean)]) / n2
covarian = (n1 * over_50K_cov + n2 * under_50K_cov) / (n1 + n2)

#w = np.transpose(((over_50K_mean - under_50K_mean)).dot(inv(covarian)) )
w = np.transpose(np.transpose(over_50K_mean - under_50K_mean).dot(inv(covarian)))
b = (-0.5) * np.transpose(over_50K_mean).dot(inv(covarian)).dot(over_50K_mean) + 0.5 * np.transpose(under_50K_mean).dot(inv(covarian)).dot(under_50K_mean) + np.log(n1 / n2)
np.save('model_w_generative', w)
np.save('model_b_generative', b)

train_z = np.dot(train_x, w) + b
train_predict = 1 / (1 + np.exp(-1 * train_z))
for i in train_predict :
    if i > 0.5 :
        i = 1
    else :
        i = 0
train_accuracy = 1 - np.sum(np.abs(train_predict - train_y)) / len(train_y)

valid_z = np.dot(valid_x, w) + b
valid_predict = 1 / (1 + np.exp(-1 * valid_z))
for i in valid_predict :
    if i > 0.5 :
        i = 1
    else :
        i = 0
valid_accuracy = 1 - np.sum(np.abs(valid_predict - valid_y)) / len(valid_y)

print(train_accuracy)
print(valid_accuracy)
