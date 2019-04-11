import numpy as np
from keras import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten,LeakyReLU, ZeroPadding2D, GaussianNoise
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
import sys, csv
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import load_model

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
            #tmp = np.reshape(flat_array, (48, 48, 1))
            #tmp = np.concatenate((tmp, tmp, tmp), axis=2)
            x_train.append(flat_array)
    y_train = np.zeros((len(y_label), 7))
    y_label = np.array(y_label)
    y_train[np.arange(y_label.shape[0]), y_label] = 1
    x_train = np.array(x_train)
    x_train = normalization(x_train)
    print(x_train.shape)
    x_train = np.reshape(x_train, (-1, 48, 48, 1))
    print(x_train.shape)
    return x_train, y_train

def split_valid(x_train, y_train, rate = 0.9) :
    size = int(x_train.shape[0] * rate)
    randomize = np.arange(len(x_train))
    np.random.shuffle(randomize)
    x_train, y_train = x_train[randomize], y_train[randomize]
    x_valid = x_train[size:]
    y_valid = y_train[size:]
    x_train = x_train[:size]
    y_train = y_train[:size]

    return x_train, y_train, x_valid, y_valid

def normalization(x) :
    x /= 255
    mean = (sum(x) / x.shape[0])
    std = np.std(x, axis=0)
    mean = np.tile(mean, (x.shape[0], 1))
    std = np.tile(std, (x.shape[0], 1))
    x = (x - mean) / std
    #x = (x - x.mean()) / (x.std())
    return x

def build_model():
    filt_size = (3, 3)
    model = Sequential()
    model.add(Convolution2D(32, filt_size, input_shape=(48,48,1), activation='relu', padding='same'))
    model.add(Convolution2D(32, filt_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(64, filt_size, activation='relu', padding='same'))
    model.add(Convolution2D(64, filt_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(128, filt_size, activation='relu', padding='same'))
    model.add(Convolution2D(128, filt_size, activation='relu', padding='same'))
    model.add(Convolution2D(128, filt_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(256, filt_size, activation='relu', padding='same'))
    model.add(Convolution2D(256, filt_size, activation='relu', padding='same'))
    #model.add(Convolution2D(256, filt_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    
    print(model.summary())
    return model

def train_model(model, x_train, y_train, x_valid, y_valid, gen = True) :
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    check  = ModelCheckpoint("./batchnormal-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1)

    if gen :
        datagen = ImageDataGenerator(
            featurewise_center=False,  
            samplewise_center=False,  
            featurewise_std_normalization=False,  
            samplewise_std_normalization=False,  
            zca_whitening=False,  
            rotation_range = 10, 
            width_shift_range=0.1, 
            height_shift_range=0.1, 
            horizontal_flip=True, 
            vertical_flip=False)

        datagen.fit(x_train)                       

        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size  = 128),
                        samples_per_epoch = x_train.shape[0],
                        epochs = 100,
                        steps_per_epoch = len(x_train),
                        validation_data = (x_valid, y_valid),
                        callbacks=[check])
    else :
        model.fit(x_train, y_train, batch_size = 128, epochs = 100, validation_data=(x_valid, y_valid), callbacks=[check, early_stopping])
    score = model.evaluate(x_valid, y_valid)
    print ('Total loss on validation data :', score[0])
    print ('Accuracy of validation data :', score[1])

    return model, history

if __name__ == "__main__":
    x_train, y_train= load_data(sys.argv[1])
    x_train, y_train, x_valid, y_valid = split_valid(x_train, y_train)
    model = build_model()
    model, history = train_model(model, x_train, y_train, x_valid, y_valid, gen=True)
    model.save('hw3_model.h5')

    print('completed')
