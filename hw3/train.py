import numpy as np
from keras import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten,LeakyReLU, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
import sys, csv
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model

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
            tmp = np.reshape(flat_array, (48, 48, 1))
            #tmp = np.concatenate((tmp, tmp, tmp), axis=2)
            x_train.append(tmp)
    y_train = np.zeros((len(y_label), 7))
    y_label = np.array(y_label)
    y_train[np.arange(y_label.shape[0]), y_label] = 1
    x_train = np.array(x_train)
    print(x_train.shape)
    return x_train, y_train

def split_valid(x_train, y_train, rate = 0.9) :
    size = int(x_train.shape[0] * rate)
    x_valid = x_train[size:]
    y_valid = y_train[size:]
    x_train = x_train[:size]
    y_train = y_train[:size]

    return x_train, y_train, x_valid, y_valid

def normalization(x) :
    x /= 255.0
    #x = (x - x.mean()) / (x.std())
    return x

def build_model():
    """model = Sequential()
    
    model.add(Convolution2D(64, (3, 3), input_shape = (48, 48, 1), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Dense(units = 7, activation = 'softmax', kernel_initializer='glorot_normal'))"""

    """model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
    
    x = Flatten()(model.output)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(7, activation='softmax')(x)
    # 重新建立模型結構
    model=Model(model.input,x)"""

    model = Sequential()
    #model.add(ZeroPadding2D((1,1),input_shape=(48, 48, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', input_shape=(48, 48, 1), padding='same'))
    model.add(BatchNormalization())
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
 
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
 
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
 
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
 
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.5))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
 
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    print(model.summary())
    return model

def train_model(model, x_train, y_train, x_valid, y_valid, gen = True) :
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    check  = ModelCheckpoint("./check/batchnormal-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1)

    if gen :
        datagen = ImageDataGenerator(
            rotation_range=0.5,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        datagen.fit(x_train, augment=True, rounds=2)
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=200), steps_per_epoch=round(len(x_train)*2/200), epochs=100, validation_data=(x_valid, y_valid), callbacks=[check, early_stopping])
    else :
        model.fit(x_train, y_train, batch_size = 100, epochs = 100, validation_data=(x_valid, y_valid), callbacks=[check, early_stopping])
    score = model.evaluate(x_valid, y_valid)
    print ('Total loss on validation data :', score[0])
    print ('Accuracy of validation data :', score[1])

    return model

def augmentation(X_train, Y_train, expand_size=5):
    datagen = ImageDataGenerator(  
        rotation_range=5,  
        width_shift_range=0,  
        height_shift_range=0,  
        shear_range=0.2,  
        zoom_range=0.2,  
        horizontal_flip=True,  
        fill_mode='nearest')

    X_aug, Y_aug = [], []
    batch_size = 32
    total = len(X_train) // batch_size + int(len(X_train) % batch_size != 0)
    for i in range(total) :
        x = X_train[batch_size * i : batch_size * (i + 1)]
        y = Y_train[batch_size * i : batch_size * (i + 1)]
        batchs = 0
        for batch_x, batch_y in datagen.flow(x, y, batch_size=32) :
            X_aug = X_aug + list(batch_x)
            Y_aug = Y_aug + list(batch_y)
            batchs += 1
            if batchs >= expand_size :
                break
    X_aug = np.array(X_aug)
    Y_aug = np.array(Y_aug)
    X_train = np.concatenate((X_train, X_aug),axis=0)
    Y_train = np.concatenate((Y_train, Y_aug),axis=0)

    randomize = np.arange(len(X_train))
    np.random.shuffle(randomize)
    X_train,Y_train = (X_train[randomize], Y_train[randomize])

    return X_train, Y_train

if __name__ == "__main__":
    x_train, y_train = load_data(sys.argv[1])
    #x_train, y_train = augmentation(x_train, y_train)
    x_train = normalization(x_train)
    x_train, y_train, x_valid, y_valid = split_valid(x_train, y_train)
    model = build_model()
    model = train_model(model, x_train, y_train, x_valid, y_valid, gen=False)
    model.save('hw3_model.h5')