import numpy as np

np.random.seed(123)

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D, Average
from keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D, Dropout
from keras.utils.data_utils import get_file
from keras.applications.resnet50 import ResNet50
from keras.utils.vis_utils import plot_model
import keras.backend as K
import tensorflow as tf

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def get_configurable_model(layers, final_activation='softmax', output_size=10):
    model = Sequential()
    for l in layers:
        model.add(l)
    model.add(Dense(output_size))
    model.add(Activation(final_activation))
    return model


def get_basic_network(input_shape, classes=10, final_activation='softmax', input=None, training=True):
    input = input if input is not None else Input(input_shape)
    layers = [
        # Conv2D(32, (3, 3), padding='valid'),
        # Conv2D(32, (3, 3), padding='valid'),
        Conv2D(64, (7, 7), padding='valid', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(128, (3, 3), padding='valid', activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='valid', activation='relu'),
        BatchNormalization(),
        # Conv2D(256, (3, 3), padding='valid', activation='relu'),
        # BatchNormalization(),
        # Conv2D(256, (3, 3), padding='same', activation='relu'),
        # BatchNormalization(),
        # MaxPooling2D(pool_size=(2, 2)),
        # Conv2D(512, (3, 3), padding='same', activation='relu'),
        # BatchNormalization(),
        # Conv2D(512, (8, 8), padding='valid', activation='relu'),
    ]

    x = input
    for l in layers:
        x = l(x)
    # x = Dense(classes)(x)
    # x = Conv2D(1024, (3, 3), padding='valid')(x)
    x = Conv2D(classes, (8, 8), padding='valid')(x)
    x = Activation(final_activation)(x)
    if training:
        x = Flatten()(x)
    return Model(input, x)


def get_all_batch_and_relu_network(input_shape, classes=10, final_activation='softmax'):
    layers = [
        Conv2D(32, (3, 3), padding='same', data_format='channels_last', input_shape=input_shape),
        # BatchNormalization(),
        Dropout(0.5),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Activation('relu'),
        GlobalAveragePooling2D(),
    ]
    return get_configurable_model(layers, output_size=classes, final_activation=final_activation)


def get_dropout_network(input_shape, classes=10, final_activation='softmax'):
    layers = [
        Conv2D(32, (3, 3), padding='same', data_format='channels_last', input_shape=input_shape),
        Conv2D(32, (3, 3), padding='same'),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        Dropout(0.25),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
    ]
    return get_configurable_model(layers, output_size=classes, final_activation=final_activation)


def get_extra_layer_network(input_shape, classes=10, final_activation='softmax'):
    layers = [
        Conv2D(32, (3, 3), padding='same', data_format='channels_last', input_shape=input_shape),
        Conv2D(32, (3, 3), padding='same'),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization()
    ]
    return get_configurable_model(layers, output_size=classes, final_activation=final_activation)


class Join(Average):
    """Layer that averages a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def build(self, input_shape):
        # print("build")
        self.shape = input_shape
        super(Join, self).build(input_shape)

    def _drop_one(self, inputs):
        rand_value = tf.random_shuffle([1.0 for _ in range(len(inputs) - 1)] + [0.0])

        output = inputs[0] * rand_value[0]
        for i in range(1, len(inputs)):
            output += inputs[i] * rand_value[i]
        return output / (len(inputs) - 1)

    def _merge_function(self, inputs):
        return K.in_train_phase(self._drop_one(inputs), super(Join, self)._merge_function(inputs))


def get_convolution_layer(input_layer, filters, kernel_size, conv_name='block'):
    layer = Conv2D(filters, kernel_size, padding='valid', name='conv-{}'.format(conv_name))(input_layer)
    layer = BatchNormalization(axis=-1, name='bn-1-{}'.format(conv_name))(layer)
    return Activation('relu', name='relu-1-{}'.format(conv_name))(layer)


def get_fractal(filters, input_layer, width=4, block_name='block', identity=False, join=True):
    kernel_size = 1 if identity else 1 + (2 ** (width - 1)) * 2
    left_filter = get_convolution_layer(input_layer, filters, (kernel_size, kernel_size),
                                        conv_name='{}-{}-a'.format(block_name, width))
    if width == 1:
        return left_filter
    else:
        fractal1 = get_fractal(filters, input_layer, width=width - 1, block_name=block_name + '1', identity=identity)
        fractal2 = get_fractal(filters, fractal1, width=width - 1, block_name=block_name + '2', identity=identity, join=False)
        outputs = [left_filter] + fractal2 if isinstance(fractal2, list) else [left_filter, fractal2]
        return Join(name='join-{}-{}'.format(block_name, width))(outputs) if join else outputs


def get_fractal_network(input_shape, classes=10, final_activation='softmax', training=True, blocks=1, width=3):
    def get_block(input_layer, filters, index):
        layers = get_fractal(filters, input_layer, width=width, block_name='block{}'.format(index))
        layers = BatchNormalization(axis=-1)(layers)
        layers = Activation('relu')(layers)
        return MaxPooling2D(pool_size=(2, 2))(layers)

    input_layer = Input(input_shape)
    x = input_layer
    x = Conv2D(64, (7, 7), padding='valid', activation='relu')(x)
    # x = BatchNormalization(axis=-1)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    for b in range(blocks):
        x = get_block(x, 128 * (2 ** b), b)

    x = AveragePooling2D()(x)
    x = Conv2D(classes, (1, 1), activation=final_activation)(x)
    if training:
        x = Flatten()(x)

    m = Model(input_layer, x, name='fractal')
    return m
