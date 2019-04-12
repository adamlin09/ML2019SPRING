import numpy as np
from keras.models import load_model
import sys, csv
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing import image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.color import label2rgb, gray2rgb ,rgb2gray
from lime import lime_image
import skimage
from keras.models import Model

model = load_model('hw3_model.h5?dl=1')
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

def normalization(x) :
    x /= 255
    mean = (sum(x) / x.shape[0])
    std = np.std(x, axis=0)
    meann = np.tile(mean, (x.shape[0], 1))
    stdd = np.tile(std, (x.shape[0], 1))
    x = (x - meann) / stdd
    #x = (x - x.mean()) / (x.std())
    return x

def pred3(inp):
    x_gray = rgb2gray(inp)
    #print(x_gray.shape)
    x_gray = x_gray.reshape(len(inp),48,48,1)
    X = np.array(x_gray)
    return model.predict(X)

def segmentation(input):
    return skimage.segmentation.slic(input)

def deprocess(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

if __name__ == "__main__":
    #model = load_model('hw3_model.h5?dl=1')
    X_train, Y_train = load_data(sys.argv[1])
    x, y = [[] for i in range(7)], [[] for i in range(7)]
    num = 0
    for i in range(len(Y_train)) :
        label = np.argmax(Y_train[i])
        if x[label] == [] :
            x[label] = X_train[i]
            y[label] = Y_train[i]
            num += 1
        else :
            continue
        if num == 7 :
            break
    x[3] = X_train[8]
    x, y = np.array(x), np.array(y)
    outpath = sys.argv[2]

    for j in range(len(x)):
        a = np.reshape(x[j], (1, 48, 48, 1))
        pred = model.predict(a)
        target_output = K.categorical_crossentropy(tf.convert_to_tensor(y, dtype=float), model.output, from_logits=False)
        last_conv_layer = model.get_layer('conv2d_1')
        grads = K.gradients(target_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([a])
        for i in range(conv_layer_output_value.shape[2]):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        heatmap = np.mean(conv_layer_output_value, axis=2)
        heatmap = np.absolute(heatmap)
        heatmap /= np.max(heatmap)
        plt.imshow(heatmap, cmap=plt.get_cmap('jet' ), interpolation='nearest')
        plt.colorbar()
    
        print(np.argmax(y[j]))
        plt.savefig(outpath + 'fig1_' + str(j) +'.jpg')
        #plt.show()
        plt.clf()

    layer_name = 'conv2d_1'
    plt.figure(figsize=(20,20), dpi = 100)
    for j in range(32):
        filter_index = j

        layer_output = model.get_layer(layer_name).output
        loss = K.mean(layer_output[:, :, :, filter_index])

        grads = K.gradients(loss, model.input)[0]

        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        iterate = K.function([model.input], [loss, grads])
        input_img_data = np.random.random((1, 48, 48, 1)) * 20 + 128.

        for i in range(50):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * 1

        img = input_img_data[0]
        img = deprocess(img)
        img = img.reshape(48,48)
        plt.subplot(4, 8, j+1)
        plt.title('filter' + str(j), fontsize=10)
        plt.imshow(img, cmap='gray')
        print(j)
    plt.title('filter of ' + layer_name, fontsize=20, y=12, x = -5)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.savefig(outpath + 'fig2_1.jpg')
    #plt.show()
    plt.clf()

    subModel = Model(model.input, model.get_layer('conv2d_1').output)
    predictions = subModel.predict(np.array([X_train[28000]]))
    w = []
    for i in range(32):
        w.append(predictions[0,:,:,i].reshape(48,48))
    plt.figure(figsize=(20,20), dpi = 100)
    for i in range(32):
        plt.subplot(4, 8, i+1)
        plt.title('filter' + str(i), fontsize=10)
        plt.imshow(w[i], cmap='gray')
        print(i)
    plt.title('output face of filter ' + layer_name + "(image = 28000)", fontsize=20, y=12, x = -5)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    plt.savefig(outpath + 'fig2_2.jpg')
    #plt.show()
    plt.clf()
    
    for i in range(len(x)) :
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
                                image=gray2rgb(x[i].reshape((48,48))), 
                                classifier_fn=pred3,
                                segmentation_fn=segmentation)
        image, mask = explanation.get_image_and_mask(
                                    label=np.argmax(y[i]),
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=5,
                                    min_weight=0.0)
        plt.imshow(np.absolute(image))
        print(i)
        plt.savefig(outpath + 'fig3_' + str(i) + '.jpg')
        #plt.show()
        plt.clf()

    print('completed !')