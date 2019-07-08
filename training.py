from data_loaders import get_svhn_data, center_data, get_non_digit_data, randomize_data, center_samples, get_svhn_test_data
from networks import get_basic_network, get_fractal_network
import numpy as np
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16
import keras.backend as K
from keras.layers import Dense, Flatten, Input, Activation, Conv2D
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import sys
from datetime import datetime


def multicategory_accuracy(y_true, y_pred):
    return K.all(K.equal(y_true, K.round(y_pred - 0.47)), axis=-1)


def weighted_binary_crossentropy(y_true, y_pred):
    return K.mean(((y_true * 5) + 1) * K.binary_crossentropy(y_true, y_pred), axis=-1)


def test_base_data(model, output_name, input_shape=None, optimizer=None, svhn_count=40*1024,
                   non_digit_count=10000000, non_digit_step_size=32, epochs=20):
    x1, y1 = get_svhn_data(label_count=11, sample_count=svhn_count)
    x2, y2 = get_non_digit_data(count=non_digit_count, step_size=non_digit_step_size)
    x_train, y_train = randomize_data(np.concatenate((x1, x2)), np.concatenate((y1, y2)))
    center_samples(x_train)

    optimizer = optimizer or SGD(lr=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    if input_shape:
        x_train = center_data(x_train, input_shape)

    d = datetime.now().strftime('%Y-%m-%dT%H%M')
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, epsilon=0.005)
    es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    log_cb = CSVLogger('reports/{}_{}.csv'.format(output_name, d))
    model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=1, validation_split=0.1,
              callbacks=[lr_cb, log_cb, es_cb])
    model.save_weights('weights/{}_{}.h5'.format(output_name, d))


def test_generated_data(model, output_name, svhn_count=40*1024, non_digit_count=10000000, input_shape=None):
    x1, y1 = get_svhn_data(label_count=10)
    x2, _ = get_non_digit_data(step_size=16)
    x_train, y_train = randomize_data(np.concatenate((x1, x2)), np.concatenate((y1, np.zeros((x2.shape[0], 10)))))

    if input_shape:
        x_train = center_data(x_train, input_shape)

    datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=20,
        # width_shift_range=0.25,
        # height_shift_range=0.25,
        validation_split=0.2,
        zoom_range=[0.5, 1],
        channel_shift_range=0.5
    )
    datagen.fit(x_train)

    model.compile(loss=weighted_binary_crossentropy,
                  optimizer=SGD(lr=0.01, momentum=0.9),
                  metrics=['binary_accuracy', multicategory_accuracy])

    d = datetime.now().strftime('%Y-%m-%dT%H%M')
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, epsilon=0.005)
    es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    log_cb = CSVLogger('reports/{}_{}.csv'.format(output_name, d))
    mc_cb = ModelCheckpoint('weights/{}_{}.h5'.format(output_name, d), save_best_only=True, save_weights_only=True)
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=128, subset='training'),
                        validation_data=datagen.flow(x_train, y_train, batch_size=128, subset='validation'),
                        epochs=50, callbacks=[lr_cb, log_cb, es_cb, mc_cb])



def run_10label_test_data(model):
    x1, y1 = get_svhn_test_data(label_count=10)
    x2, y2 = get_non_digit_data(test=True)
    x_train, y_train = np.concatenate((x1, x2)), np.concatenate((y1, np.zeros((x2.shape[0], 10))))
    center_samples(x_train)

    model.compile(loss=weighted_binary_crossentropy,
                  optimizer=SGD(lr=0.01, momentum=0.9),
                  metrics=['binary_accuracy', multicategory_accuracy])

    print(model.evaluate(x_train, y_train, batch_size=128))


def train_basic():
    model = get_basic_network(input_shape=(32, 32, 3), classes=11)
    model.summary()
    test_base_data(model, 'basic')


def train_basic10():
    model = get_basic_network((32, 32, 3), classes=10, final_activation='sigmoid')
    model.summary()
    test_generated_data(model, 'basic10')


def train_weighted_vgg():
    input_img = Input(shape=(64, 64, 3))
    model = VGG16(classes=11, input_tensor=input_img, include_top=False) # get_vgg(x.shape[1:], classes=11)
    # for layer in model.layers:
    #     layer.trainable = False
    x = Flatten(name='flatten')(model.layers[-1].output)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(11, activation='softmax', name='predictions')(x)

    model = Model(input_img, x)

    test_base_data(model, 'weighted_vgg', input_shape=(64, 64, 3), optimizer=Adam(lr=0.0001),
                   non_digit_count=5000, svhn_count=60*1024)


def train_vgg():
    input_img = Input(shape=(64, 64, 3))
    model = VGG16(classes=11, input_tensor=input_img, include_top=False, weights=None)
    x = Flatten(name='flatten')(model.layers[-1].output)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(11, activation='softmax', name='predictions')(x)

    model = Model(input_img, x)

    test_base_data(model, 'vgg', input_shape=(64, 64, 3), optimizer=Adam(lr=0.00005), non_digit_count=5000,
                   svhn_count=60*1024)


def train_fractal():
    model = get_fractal_network(input_shape=(32, 32, 3), classes=11)
    model.summary()
    test_base_data(model, 'fractal')


def train_fractal_all():
    model = get_fractal_network(input_shape=(32, 32, 3), classes=11)
    x1, y1 = get_svhn_data(label_count=11)
    x2, y2 = get_non_digit_data(step_size=16)
    x_train, y_train = randomize_data(np.concatenate((x1, x2)), np.concatenate((y1, y2)))

    datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True,
        rotation_range=20,
        # width_shift_range=0.25,
        # height_shift_range=0.25,
        validation_split=0.2,
        zoom_range=[0.5, 1],
        channel_shift_range=0.5
    )
    datagen.fit(x_train)
    d = datetime.now().strftime('%Y-%m-%dT%H%M')
    lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, epsilon=0.005)
    es_cb = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    log_cb = CSVLogger('reports/fractallall_{}.csv'.format(d))
    mc_cb = ModelCheckpoint('weights/fractallall_{}.h5'.format(d), save_best_only=True, save_weights_only=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, momentum=0.9),
                  metrics=['accuracy'])
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=128, subset='training'),
                        validation_data=datagen.flow(x_train, y_train, batch_size=128, subset='validation'),
                        epochs=50, callbacks=[lr_cb, log_cb, es_cb, mc_cb])

    test_base_data(model, 'fractal', non_digit_step_size=16, epochs=50)


def train_fractal10():
    model = get_fractal_network(input_shape=(32, 32, 3), classes=10)  # get_vgg(x.shape[1:], classes=11)  # get_vgg(x.shape[1:], classes=11)
    test_generated_data(model, 'fractal10')


def train_fractal_test():
    model = get_fractal_network(input_shape=(32, 32, 3), classes=10)  # get_vgg(x.shape[1:], classes=11)
    model.load_weights('weights/recursive10_2018-04-22T2342.h5')
    run_10label_test_data(model)


if __name__ == "__main__":
    m = sys.argv[1]
    locals()['train_' + sys.argv[1]]()
