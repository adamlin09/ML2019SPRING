import numpy as np
from keras import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model
import sys, csv
from train import normalization

def load_data(input) :
    x_test = []
    data = csv.reader(open(input))
    num = 0
    for row in data :
        if num == 0 :
            num += 1
            continue
        else :
            flat_array = np.array(row[1].split(' '), dtype = float)
            #tmp = np.reshape(flat_array, (48, 48, 1))
            #tmp = np.concatenate((tmp, tmp, tmp), axis=2)
            x_test.append(flat_array)
    
    x_test = np.array(x_test)
    x_test = normalization(x_test)
    x_test = np.reshape(x_test, (-1, 48, 48, 1))
    return x_test

def test(x_test, modelfile) :
    model = load_model(modelfile)
    result = model.predict(x_test)
    result = np.argmax(result, axis=1)
    return result

def write(filename, data) :
    outfile = open(filename, 'w')
    writer = csv.writer(outfile)
    writer.writerow(['id', 'label'])
    for i in range(data.shape[0]) :
        id = i
        label = data[i]
        writer.writerow([id, label])
    outfile.close()

if __name__ == "__main__":
    modelfile = './hw3_model.h5?dl=1'
    x_test = load_data(sys.argv[1])
    #x_test = normalization(x_test)
    result = test(x_test, modelfile)
    write(sys.argv[2], result)