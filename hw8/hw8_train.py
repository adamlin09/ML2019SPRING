from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
import sys, csv, os, time
import warnings

from keras.preprocessing import image

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Conv2D
from keras.layers import Dense, Flatten, LeakyReLU, AveragePooling2D, MaxPool2D
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.utils import conv_utils
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.engine import InputSpec
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

start = time.time()

BASE_WEIGHT_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.6/'

# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config = config)

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

def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


class DepthwiseConv2D(Conv2D):
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config


def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000):

    if K.backend() != 'tensorflow':
        raise RuntimeError('Only Tensorflow backend is currently supported, '
                           'as other backends do not support '
                           'depthwise convolution.')

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape.
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=None)
    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        if rows != cols or rows not in [128, 160, 192, 224]:
            raise ValueError('If imagenet weights are being loaded, '
                             'input must have a static square shape (one of '
                             '(128,128), (160,160), (192,192), or (224, 224)).'
                             ' Input shape provided = %s' % (input_shape,))

    if K.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = _conv_block(img_input, 8, alpha)
    x = _depthwise_conv_block(x, 8, alpha, depth_multiplier, block_id=1)

    x = _conv_block(x, 16, alpha)
    x = _depthwise_conv_block(x, 16, alpha, depth_multiplier, block_id=3)

    x = MaxPool2D(pool_size=(2, 2), padding="same")(x)

    x = _conv_block(x, 32, alpha)
    x = _depthwise_conv_block(x, 32, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 32, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    
    x = MaxPool2D(pool_size=(2, 2), padding="same")(x)
    x = _conv_block(x, 64, alpha)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=9)
    x = MaxPool2D(pool_size=(2, 2), padding="same")(x)
    # x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=10)
    # x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=11)

    # x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
    #                           strides=(2, 2), block_id=12)
    # x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    if pooling == 'avg':
        x = AveragePooling2D(padding="same")(x)
    elif pooling == 'max':
        x = MaxPool2D(padding="same")(x)
            
    x = Flatten()(x)
    x = Dense(30)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)
    x = Dense(7)(x)
    x = Activation('softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_last" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')
        model.load_weights(weights_path)

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

def train_model(model, x_train, y_train, x_valid, y_valid, gen = True) :
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    if not os.path.exists('./check'):
        os.makedirs('./check')

    check  = ModelCheckpoint("./check/batchnormal-{epoch:05d}-{val_acc:.5f}.h5",monitor='val_acc',save_best_only=True, verbose=1, save_weights_only=True)
    # early_stopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1)

    if gen :
        datagen = ImageDataGenerator(
            featurewise_center=False,  
            samplewise_center=False,  
            featurewise_std_normalization=False,  
            samplewise_std_normalization=False,  
            zca_whitening=False,  
            rotation_range = 15, 
            width_shift_range=0.2, 
            height_shift_range=0.2, 
            horizontal_flip=True, 
            vertical_flip=False)

        datagen.fit(x_train)
        
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size  = 128),
                        epochs = 100,
                        steps_per_epoch = len(x_train)/128,
                        validation_data = (x_valid, y_valid),
                        callbacks=[check])
    else :
        history = model.fit(x_train, y_train, batch_size = 128, epochs = 100, validation_data=(x_valid, y_valid), callbacks=[check])
    score = model.evaluate(x_valid, y_valid)
    print ('Total loss on validation data :', score[0])
    print ('Accuracy of validation data :', score[1])

    return model, history

if __name__ == '__main__':
    model = MobileNet(include_top=False, weights=None, input_shape=(48, 48, 1), alpha=1, classes=7, pooling='avg')
    model.summary()

    x_train, y_train= load_data(sys.argv[1])
    x_train, y_train, x_valid, y_valid = split_valid(x_train, y_train)
    np.save('x_valid.npy', x_valid)
    np.save('y_valid.npy', y_valid)

    model, history = train_model(model, x_train, y_train, x_valid, y_valid, gen=True)
    model.save_weights('weightsfile.h5')
    
end = time.time()
elapsed = end - start
print('')
print("Time taken: ", elapsed, "seconds.")