import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import Adam, SGD
import h5py
import numpy as np
from scipy.io import loadmat
from networks import get_basic_network
from keras.layers import Input
r = loadmat('/Users/mikehenderson/Downloads/train_32x32.mat')

images = []
for i in range(r['X'].shape[3]):
    image = r['X'][:,:,:,i].astype('float')
    image[:,:,0] -= image[:,:,0].mean()
    image[:,:,1] -= image[:,:,1].mean()
    image[:,:,2] -= image[:,:,2].mean()
    images.append(image)
images = np.asarray(images)
labels = r['y']
labels = np.where(labels == 10, 0, labels)
labels = np_utils.to_categorical(labels, num_classes=10)
# filepath = '/Users/mikehenderson/Downloads/train/digitStruct.mat'
# arrays = {}
# f = h5py.File(filepath)
# print(f['digitStruct'])
# print([x for x in f[f['digitStruct'].items()[0][1][0][0]].items()])
# print(f['digitStruct'].items()[1][1])

x_train = images[:50000]
y_train = labels[:50000]
x_test = images[50000:]
y_test = labels[50000:]


batch_size = 128
learning_rate = 0.001
training_size = 1024 * 8
test_size = 4096


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', data_format='channels_last', input_shape=x_train.shape[1:]))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
# model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
# model.add(Dropout(0.25))
model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

def get_configurable_model(layers, final_activation='sigmoid'):
    model = Sequential()
    for l in layers:
        model.add(l)
    model.add(Dense(10, activation=final_activation))
    return model

def get_accuracy(y_true, y_pred):
    zeros = np.zeros(y_pred.shape)
    correct = np.argmax(y_pred, y_true, 0).sum()
    # ones = np.ones(y_pred.shape)
    # same = np.where(np.where(y_pred > 0.75, ones, zeros) == y_true, ones, zeros)
    # correct = np.all(same, axis=1).sum()
    return correct * 1.0 / y_pred.shape[0]

# print(keras.backend.sum(keras.backend.cast(keras.backend.all(keras.backend.equal(y_true, y_pred), axis=1), 'float32')))
# input = Input((32, 32, 3))
# model = get_basic_network(None, input=input)
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=learning_rate, momentum=0.9),
              metrics=['accuracy'])

# # 9. Fit model on training data
# # model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=1)
for iteration in range(1):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1, validation_split=0.1)
    # results = model.predict(x_train)
    # print('Train: ', get_accuracy(y_train, results))
    # print('Test: ', get_accuracy(y_test, model.predict(x_test)))


# 10. Evaluate model on test data
# print(model.evaluate(x_test, y_test, verbose=0))
