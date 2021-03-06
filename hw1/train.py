import csv, sys, math
import numpy as np

data = [[] for i in range(18)]
f = open('./ml2019spring-hw1/train.csv', encoding = 'Big5')
d = csv.reader(f)
index = 0
for i in d :
	if index == 0 :
		pass
	else :
		for j in i[3:] :
			if j == 'NR' :
				data[(index-1)%18].append(0.0)
			else:
				a = float(j)
				if a < 0.0 :
					a = -a
				data[(index-1)%18].append(float(j))
	index += 1
f.close()

train_x, train_y = [], []
for i in range(12) :
	for j in range(471):
		train_x.append([])
		for k in range(18) :
			if k == 13 or k == 1 or k == 10:
				continue
			for l in range(9) :
				train_x[471*i+j].append(data[k][480*i+j+l])
		train_y.append(data[9][480*i+9+j])
train_x = np.array(train_x)
train_y = np.array(train_y)
train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x), axis=1)

#initial
w = np.ones(train_x.shape[1])
lr = 0.5
iteration = 300000
prev_grad = 0

for i in range(iteration) :
	y = np.dot(train_x, w)
	loss = y - train_y
	grad = 2*np.dot(train_x.T, loss)
	prev_grad += (grad**2)
	ada = np.sqrt(prev_grad)
	w -= lr * grad / ada
	if i % 1000 == 0 :
		cost = np.sum(loss**2) / len(loss)
		cost_a = math.sqrt(cost)
		print(cost_a)

np.save('model.npy', w)