import numpy as np
from keras import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, Flatten,LeakyReLU
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
import sys, csv

def load_data(input) :
    x_train, y_label = [], []
    train_data = csv.reader(open(input))
    num = 0
    for row in train_data :
        if num == 0 :
            num += 1
            continue
        else :
            y_label.append(int(row[0]))
            flat_array = np.array(row[1].split(' '), dtype = float)
            x_train.append(np.reshape(flat_array, (48, 48, 1)))
    y_train = np.zeros((len(y_label), 7))
    y_label = np.array(y_label)
    y_train[np.arange(y_label.shape[0]), y_label] = 1
    x_train = np.array(x_train)
    return x_train, y_train

def split_valid(x_train, y_train, rate = 0.9) :
    size = int(x_train.shape[0] * rate)
    x_valid = x_train[size:]
    y_valid = y_train[size:]
    x_train = x_train[:size]
    y_train = y_train[:size]

    return x_train, y_train, x_valid, y_valid

def normalization(x) :
    #x /= 255
    x = (x - x.mean()) / x.std()
    return x

def build_model():
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), input_shape = (48, 48, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(units = 512, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))

    model.add(Dense(units = 512, kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    
    model.add(Dense(units = 7, activation = 'softmax', kernel_initializer='glorot_normal'))

    return model

def train_model(model, x_train, y_train, x_valid, y_valid) :
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    check  = ModelCheckpoint("./check/batchnormal-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1)

    model.fit(x_train, y_train, batch_size = 100, epochs = 40, validation_data=(x_valid, y_valid), callbacks=[check, early_stopping])
    score = model.evaluate(x_valid, y_valid)
    print ('Total loss on validation data :', score[0])
    print ('Accuracy of validation data :', score[1])

    return model

if __name__ == "__main__":
    x_train, y_train = load_data(sys.argv[1])
    x_train = normalization(x_train)
    x_train, y_train, x_valid, y_valid = split_valid(x_train, y_train)
    model = build_model()
    model = train_model(model, x_train, y_train, x_valid, y_valid)
    model.save('hw3_model.h5')