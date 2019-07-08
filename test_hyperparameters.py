from data_loaders import get_svhn_data, center_samples
from networks import get_fractal_network
from keras.optimizers import Adam, SGD
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
import sys
from datetime import datetime


def multicategory_accuracy(y_true, y_pred):
    return K.all(K.equal(y_true, K.round(y_pred - 0.47)), axis=-1)


def weighted_binary_crossentropy(y_true, y_pred):
    return K.mean(((y_true * 5) + 1) * K.binary_crossentropy(y_true, y_pred), axis=-1)


def test_data(model, output_name, batch_size=128, validation_split=0.0,):
    x_train, y_train = get_svhn_data(label_count=10, sample_count=20*1024)
    center_samples(x_train)

    d = datetime.now().strftime('%Y-%m-%dT%H%M')
    callbacks = [
        CSVLogger('reports/hyperparams/{}_{}.csv'.format(output_name, d))
    ]
    model.fit(x_train, y_train, batch_size=batch_size, epochs=20, verbose=1, callbacks=callbacks,
              validation_split=validation_split)


def test_batch_size():
    for batch_size in [32, 64, 128, 256]:
        model = get_fractal_network(input_shape=(32, 32, 3))
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.01, momentum=0.9),
                      metrics=['accuracy'])

        test_data(model, 'batchsize{}'.format(batch_size), batch_size=batch_size)


def test_learning_rate():
    for learning_rate in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        model = get_fractal_network(input_shape=(32, 32, 3))
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=learning_rate, momentum=0.9),
                      metrics=['accuracy'])

        test_data(model, 'learningrate{}'.format(learning_rate))


def test_sgd_vs_adam():
    model = get_fractal_network(input_shape=(32, 32, 3))
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, momentum=0.9),
                  metrics=['accuracy'])

    test_data(model, 'sgd', validation_split=0.2)

    model = get_fractal_network(input_shape=(32, 32, 3))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.01),
                  metrics=['accuracy'])
    test_data(model, 'adam', validation_split=0.2)


if __name__ == "__main__":
    m = sys.argv[1]
    locals()['test_' + sys.argv[1]]()
