import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import Adam
import h5py
import numpy as np
from scipy.io import loadmat
r = loadmat('/Users/mikehenderson/Downloads/train_32x32.mat')

images = []
for i in range(r['X'].shape[3]):
    images.append(r['X'][:,:,:,i])
images = np.asarray(images)
print(images.shape)
# filepath = '/Users/mikehenderson/Downloads/train/digitStruct.mat'
# arrays = {}
# f = h5py.File(filepath)
# print(f['digitStruct'])
# print([x for x in f[f['digitStruct'].items()[0][1][0][0]].items()])
# print(f['digitStruct'].items()[1][1])



batch_size = 128
learning_rate = 0.0001
training_size = 1024 * 8
test_size = None

def merge_sets(x, y, size=None):
    return np.hstack((x[::2], x[1::2]))[:size], (y[::2] + y[1::2])[:size]

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
x_train, y_train = merge_sets(x_train, y_train, size=training_size)
x_test, y_test = merge_sets(x_test, y_test, size=test_size)
x_train = x_train.astype('float') - x_train.mean()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.astype('float') - x_test.mean()
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', data_format='channels_last', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
# model.add(Dropout(0.5))
model.add(Dense(10, activation='sigmoid'))

def get_configurable_model(layers, final_activation='sigmoid'):
    model = Sequential()
    for l in layers:
        model.add(l)
    model.add(Dense(10, activation=final_activation))
    return model

def get_multi_class_accuracy(y_true, y_pred):
    zeros = np.zeros(y_pred.shape)
    ones = np.ones(y_pred.shape)
    same = np.where(np.where(y_pred > 0.75, ones, zeros) == y_true, ones, zeros)
    correct = np.all(same, axis=1).sum()
    return correct * 1.0 / y_pred.shape[0]

# print(keras.backend.sum(keras.backend.cast(keras.backend.all(keras.backend.equal(y_true, y_pred), axis=1), 'float32')))

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=learning_rate),
              metrics=['accuracy'])

# 9. Fit model on training data
# model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=1)
for iteration in range(3):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
    results = model.predict(x_train)
    print('Train: ', get_multi_class_accuracy(y_train, results))
    print('Test: ', get_multi_class_accuracy(y_test, model.predict(x_test)))


# 10. Evaluate model on test data
# print(model.evaluate(x_test, y_test, verbose=0))
