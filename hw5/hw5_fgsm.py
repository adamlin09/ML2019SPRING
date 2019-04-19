import numpy as np
from keras.applications import vgg16, vgg19
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
import glob, sys
import keras.backend as K
from keras_applications.resnet import ResNet101, ResNet152
import keras

input_dir = sys.argv[1]
f_names = glob.glob(input_dir + '*.png')

imgs = []
for i in range(len(f_names)):  # f_names为所有图片地址，list
    img = image.load_img(input_dir + str(i).zfill(3) + '.png', target_size=(224, 224))  # 读取图片
    #img.show()
    arr_img = image.img_to_array(img)  # 图片转换为数组
    arr_img = np.expand_dims(arr_img, axis=0)   # 增加第一个batch维度
    imgs.append(arr_img) # 把图片数组加到一个列表里面
    print("loading no.%s image."%i)
x = np.concatenate([i for i in imgs]) # 把所有图片数组concatenate在一起
print(x.shape)


print("predicting...")
model = vgg19.VGG19(weights='imagenet', backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)

for i in range(x.shape[0]) :
    x_adv = np.reshape(x[i], (1, 224, 224, 3))
    y = model.predict(x_adv)
    label = np.argmax(y)

    sess = K.get_session()
    
    e = 10
    print('saving...' + str(i).zfill(3))

    # One hot encode the initial class
    target = K.one_hot(label, 1000)
    
    # Get the loss and gradient of the loss wrt the inputs
    loss = K.categorical_crossentropy(target, model.output)
    grads = K.gradients(loss, model.input)
    delta = K.sign(grads[0])
    x_adv = x_adv + e * delta
    x_adv = sess.run(x_adv, feed_dict={model.input:np.reshape(x[i], (1, 224, 224, 3))})

    
    img = image.array_to_img(x_adv[0])
    if not os.path.exists(sys.argv[2]) :
        os.makedirs(sys.argv[2])
    out_path = sys.argv[2] + str(i).zfill(3) + '.png'
    img.save(out_path)
print("Completed!")