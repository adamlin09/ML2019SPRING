import csv, sys
import numpy as np
import math

w = np.load('model_w_generative.npy')
b = np.load('model_b_generative.npy')
X_test = open(sys.argv[1])
testfile = csv.reader(X_test)
test_x = []
test_y = []

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

test_x = np.array(test_x)
#normalization
#mean = np.load('mean.npy')
#std = np.load('std.npy')
mean = np.load('mean.npy')
std = np.load('std.npy')
for i in range(test_x.shape[0]) :
    for j in range(test_x.shape[1]) :
        if std[j] != 0 :
            test_x[i][j] = (test_x[i][j] - mean[j]) / std[j]

z = np.dot(test_x, w)
test_y = 1 / (1 + np.exp(-z))
test_y = np.clip(test_y, 1e-6, 1 - 1e-6)

outfile = open(sys.argv[2], 'w')
writer = csv.writer(outfile)
writer.writerow(['id', 'label'])
for i in range(len(test_y)) :
    id = i + 1
    label = 0
    if test_y[i] > 0.5 :
        label = 1
    writer.writerow([id, label])
outfile.close()