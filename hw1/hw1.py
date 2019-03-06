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
    elif row % 18 == 13 or row % 18 == 1 or row % 18 == 10:
        row += 1
        continue
    for j in range(9) :
        if i[2+j] == 'NR' :
            test_x[row//18].append(0.0)
        else :
            a = float(i[2+j])
            if a < 0.0 :
                a = -a
            test_x[row//18].append(a)
    row += 1
test_y = np.dot(test_x, w)

outfile = open(sys.argv[2], 'w')
writer = csv.writer(outfile)
writer.writerow(['id', 'value'])
for i in range(len(test_y)) :
    id = 'id_' + str(i)
    writer.writerow([id, test_y[i]])
outfile.close()