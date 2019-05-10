import jieba
from gensim.models import Word2Vec
import sys, os, csv
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten,LeakyReLU, ZeroPadding2D, GaussianNoise
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
import sys, csv, os, time
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import load_model
from keras.layers import GRU, LSTM, Bidirectional

start = time.time()
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config = config)

f = pd.read_csv(sys.argv[1])
test_x = f['comment']
test_x = np.array(test_x)
for i in range(test_x.shape[0]):
    test_x[i] = jieba.lcut(test_x[i])
test_x = np.array(test_x)

w2vmodel = Word2Vec.load('word2vec.model')
tx2 = np.zeros((len(test_x), 40, 250))
for i in range(len(test_x)):
    # x = np.zeros((40, 250))
    if len(test_x[i]) > 40:
        for j in range(40):
            try:
                tx2[i][j] = w2vmodel[test_x[i][j]]
            except KeyError as e:
                pass
    else:
        for j in range(len(test_x[i])):
            try:
                a = w2vmodel[test_x[i][j]]
                tx2[i][j] = a
            except KeyError as e:
                pass
np.save('tx2.npy', tx2)

outputs, models = [], []
model0 = load_model('./check0/batchnormal-00006-0.75179.h5')
models.append(model0)
model1 = load_model('./check1/batchnormal-00006-0.75185.h5')
models.append(model1)
# model2 = load_model('./check2/batchnormal-00004-0.74332.h5')
# models.append(model2)
model3 = load_model('./check3/batchnormal-00006-0.74938.h5')
models.append(model3)
model4 = load_model('./check4/batchnormal-00004-0.74909.h5')
models.append(model4)
model5 = load_model('./check5/batchnormal-00005-0.74821.h5')
models.append(model5)
# model6 = load_model('./check6/batchnormal-00006-0.74597.h5')
# models.append(model6)
for i in range(len(models)):
    name = 'model_' + str(i) + '.h5'
    model = models[i]
    print(name + ' loaded')

    output = model.predict(tx2.reshape((-1, 40, 250)))
    outputs.append(output)

outfile = open(sys.argv[3], 'w')
writer = csv.writer(outfile)
writer.writerow(['id', 'label'])
for i in range(len(test_x)):
    print('\r>row: {}'.format(i), end="", flush=True)
    # output = model.predict(tx2[i].reshape((-1, 40, 250)))
    num0, num1 = 0, 0
    for j in range(len(models)):
        k = np.argmax(outputs[j][i])
        if k == 0:
            num0 += 1
        else:
            num1 += 1
    if num0 > num1:
        writer.writerow([i, 0])
    else:
        writer.writerow([i, 1])
outfile.close()

end = time.time()
elapsed = end - start
print('')
print("Time taken: ", elapsed, "seconds.")