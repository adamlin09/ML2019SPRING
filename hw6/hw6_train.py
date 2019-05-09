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
import sys, csv, os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import load_model
from keras.layers import GRU, LSTM, Bidirectional

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config = config)

jieba.load_userdict(sys.argv[4])

f = pd.read_csv(sys.argv[1])
train_x = f['comment'][:119018]
f = pd.read_csv(sys.argv[2])
train_y = f['label'][:119018]
f = pd.read_csv(sys.argv[3])
test_x = f['comment']

train_x, train_y, test_x = np.array(train_x), np.array(train_y), np.array(test_x)
for i in range(train_x.shape[0]):
    train_x[i] = jieba.lcut(train_x[i])
print('1')
for i in range(test_x.shape[0]):
    test_x[i] = jieba.lcut(test_x[i])
print('2')
train_x, train_y, test_x = np.array(train_x), np.array(train_y), np.array(test_x)
np.save('train_x.npy', train_x)
np.save('train_y.npy', train_y)
np.save('test_x.npy', test_x)
# train_x = np.load('train_x.npy')
# train_y = np.load('train_y.npy')
# test_x = np.load('test_x.npy')
x_all = np.concatenate((train_x, test_x))
print(x_all[0])
label_y = np.zeros((len(train_y), 2))
label_y[np.arange(train_y.shape[0]), train_y] = 1

w2vmodel = Word2Vec(x_all, size=250, min_count=3, workers=4)
w2vmodel.save("word2vec.model")
# w2vmodel = Word2Vec.load('word2vec.model')
tx1 = np.zeros((len(train_x), 40, 250))
tx2 = np.zeros((len(test_x), 40, 250))
for i in range(len(train_x)):
    # x = np.zeros((40, 250))
    if len(train_x[i]) > 40:
        for j in range(40):
            try:
                tx1[i][j] = w2vmodel[train_x[i][j]]
            except KeyError as e:
                pass
    else:
        for j in range(len(train_x[i])):
            try:
                a = w2vmodel[train_x[i][j]]
                tx1[i][j] = a
            except KeyError as e:
                pass
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
train_data = [[] for i in range(7)]
labels = [[] for i in range(7)]
s = len(tx1) // 7
for i in range(7):
    train_data[i] = tx1[s*i : len(tx1) if s*(i+1) > len(tx1) else s*(i+1)]
    labels[i] = label_y[s*i : len(tx1) if s*(i+1) > len(tx1) else s*(i+1)]
# np.save('tx1.npy', tx1)
# np.save('tx2.npy', tx2)
# print(tx1.shape)
# print(tx2.shape)
# print(tx1[0])
# print(tx2[0])

def build_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(256, recurrent_dropout=0.2, dropout=0.2, return_sequences=True), input_shape=(40, 250)))
    model.add(Bidirectional(LSTM(256, recurrent_dropout=0.2, dropout=0.2, return_sequences=True), input_shape=(40, 250)))
    # model.add(Bidirectional(LSTM(256, recurrent_dropout=0.2, dropout=0.2, return_sequences=True), input_shape=(40, 250)))
    # model.add(Bidirectional(LSTM(512, recurrent_dropout=0.2, dropout=0.2, return_sequences=True), input_shape=(40, 250)))
    model.add(Bidirectional(LSTM(256, recurrent_dropout=0.2, dropout=0.2, return_sequences=False), input_shape=(40, 250)))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(2, activation='softmax'))
    model.summary()
    return model

for i in range(7):
    train = 0
    label = 0
    if i == 0:
        train = train_data[1]
        label = labels[1]
        for j in range(2,7):
            if j != i:
                train = np.concatenate((train, train_data[j]))
                label = np.concatenate((label, labels[j]))
    else:
        train = train_data[0]
        label = labels[0]
        for j in range(1,7):
            if j != i:
                train = np.concatenate((train, train_data[j]))
                label = np.concatenate((label, labels[j]))
    if not os.path.exists('./check' + str(i)):
        os.makedirs('./check' + str(i))
    check  = ModelCheckpoint("./check" + str(i) + "/batchnormal-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True, verbose=1)
    model = build_model()
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(train, label, batch_size=100, epochs=15, validation_data=(train_data[i], labels[i]), verbose=1, callbacks=[check])
    name = 'model_' + str(i) + '.h5'
    model.save(name)
    print(name + 'saved')
# model = load_model('./check/batchnormal-00005-0.75105.h5')
