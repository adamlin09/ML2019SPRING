import csv, sys
import numpy as np

w = np.load('model.npy')
testfile = open(sys.argv[1])
testdata = csv.reader(testfile)
row = 0
test_x = []
test_y = []

for i in testdata :
    if row % 18 == 0 :
        test_x.append([1])
    for j in range(9) :
        test_x[row//18].append(0.0 if i[2+j] == 'NR' else float(i[2+j]))
    row += 1
test_y = np.dot(test_x, w)

outfile = open(sys.argv[2], 'w')
writer = csv.writer(outfile)
writer.writerow(['id', 'value'])
for i in range(len(test_y)) :
    id = 'id_' + str(i)
    writer.writerow([id, test_y[i]])
outfile.close()