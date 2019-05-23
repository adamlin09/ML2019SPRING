import numpy as np
from keras import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten,LeakyReLU, ZeroPadding2D, GaussianNoise, UpSampling2D, Conv2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
import sys, csv, glob, time
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.models import load_model
from skimage import io, img_as_float64, img_as_float, img_as_int
from keras.layers import Dense, Input
from keras.models import Model
from keras import backend as K
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.manifold import TSNE
import pandas as pd

start = time.time()

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config = config)

def load_data(path):
    f_names = sorted(glob.glob(path + '*.jpg'))

    imgs = []
    for i in range(len(f_names)):  # f_names为所有图片地址，list
        img = io.imread(f_names[i]).astype('float')
        img /= 255.0
        imgs.append(img) # 把图片数组加到一个列表里面
        print('\r> Loading \'{}\''.format(f_names[i]), end="", flush=True)
    print('')
    imgs = np.array(imgs)
    return imgs

# input_img = Input(shape=(32,32,3))  # adapt this if using `channels_first` image data format

# x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2), padding='same')(x)
# x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# encoded = MaxPooling2D((2, 2), padding='same')(x)

# # at this point the representation is (4, 4, 8) i.e. 128-dimensional

# x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
# x = BatchNormalization()(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = UpSampling2D((2, 2))(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
# x = BatchNormalization()(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# autoencoder = Model(input_img, decoded)
# autoencoder.summary()
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# train_data = np.load('train_data.npy')
train_data = load_data(sys.argv[1])
np.save('train_data.npy', train_data)
# print(train_data[0])
# print(img_as_float(train_data[0]))
# print(img_as_float(train_data[0])*255.0)
# train_data /= 255.0

# autoencoder.fit(train_data, train_data,
#                 epochs=50,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_split=0.1)
# encoder = Model(input_img, encoded)
# encoder.save('encoder.h5')
# autoencoder.save('autoencoder.h5')
encoder = load_model('encoder.h5')
# autoencoder = load_model('autoencoder.h5')
encoded_imgs = encoder.predict(train_data)
print(encoded_imgs.shape)

pca = PCA(n_components=200, copy=False, whiten=True, svd_solver='full')
new_vec = pca.fit_transform(encoded_imgs.reshape(-1, 256))
# tsne = TSNE(n_components=4, init='pca', random_state=0)
# new_vec = tsne.fit_transform(encoded_imgs.reshape(-1, 256))

# test = autoencoder.predict(train_data[6].reshape(-1, 32, 32, 3))
# t = test*255.0
# io.imsave('test.jpg', t[0].astype('uint8'))
# print('OK')

kmeans = KMeans(n_clusters=2, random_state=0)
kmeans = kmeans.fit(new_vec)

f = pd.read_csv(sys.argv[2])
name1 = f['image1_name']
name2 = f['image2_name']
outfile = open(sys.argv[3], 'w')
writer = csv.writer(outfile)
writer.writerow(['id', 'label'])
for i in range(len(name1)):
    print('\r>id: {}'.format(i), end="", flush=True)
    if kmeans.labels_[name1[i]-1] == kmeans.labels_[name2[i]-1]:
        writer.writerow([i, 1])
    else:
        writer.writerow([i, 0])
outfile.close()
print('')

end = time.time()
elapsed = end - start
print('')
print("Time taken: ", elapsed, "seconds.")