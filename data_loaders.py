import numpy as np
np.random.seed(123)
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
from scipy.io import loadmat
import urllib2
import cv2
from keras.metrics import top_k_categorical_accuracy
import keras.backend as K


def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train.astype('float') - x_train.mean()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.astype('float') - x_test.mean()
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    return x_train, y_train, x_test, y_test


def download_file_if_not_exists(filename, url):
    if os.path.exists(filename):
        return

    with open(filename, 'wb') as f:
        f.write(urllib2.urlopen(url).read())
        f.close()


def center_data(x, new_shape):
    nx = np.zeros((x.shape[0], new_shape[1], new_shape[0], x.shape[3]))
    ux = int(new_shape[1] / 2 - x.shape[1] / 2)
    uy = int(new_shape[0] / 2 - x.shape[2] / 2)
    nx[:, uy:uy+x.shape[1], ux:ux+x.shape[2],:] = x
    return nx


def resize_labels(y, size):
    ny = np.zeros((y.shape[0], size))
    ny[:,:y.shape[1]] = y
    return ny


def get_no_numbers(test=False, step_size=32):
    imagesFiles = sorted([f for f in os.listdir('input') if f.startswith('test-no-numbers' if test else 'no-numbers')])
    files = [os.path.join('input', f) for f in imagesFiles]
    images = []
    for p in files:
        f = cv2.imread(p, cv2.IMREAD_COLOR)
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        for i in range(0, f.shape[0], step_size):
            if i + 32 > f.shape[0]:
                break
            for j in range(0, f.shape[1], step_size):
                if j + 32 > f.shape[1]:
                    break

                images.append(f[i:i+32,j:j+32].astype('float'))
    return images


def get_non_digit_data(count=1000000, test=False, step_size=32):
    init_images = get_no_numbers(test=test, step_size=step_size)[:count]
    return np.asarray(init_images), np_utils.to_categorical(np.ones((len(init_images), 1)) * 10, 11)


def get_svhn_data(sample_count=None, label_count=11):
    download_file_if_not_exists('train_32x32.mat', 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat')
    sample_count = sample_count or 10000000
    init_images = []
    r = loadmat('train_32x32.mat')
    init_labels = r['y'][:sample_count]
    init_labels = np.where(init_labels == 10, 0, init_labels)

    for i in range(min(r['X'].shape[3], sample_count)):
        image = r['X'][:, :, :, i]
        init_images.append(image.astype('float'))

    return np.asarray(init_images), np_utils.to_categorical(init_labels, label_count)


def get_svhn_test_data(label_count=11):
    download_file_if_not_exists('test_32x32.mat', 'http://ufldl.stanford.edu/housenumbers/test.tar.gz')
    r = loadmat('train_32x32.mat')
    init_images = []
    init_labels = r['y'][:20000]
    init_labels = np.where(init_labels == 10, 0, init_labels)

    for i in range(len(init_labels)):
        image = r['X'][:, :, :, i]
        init_images.append(image.astype('float'))

    return np.asarray(init_images), np_utils.to_categorical(init_labels, label_count)


def randomize_data(images, labels):
    r = np.random.permutation(len(images))
    images2 = []
    labels2 = []

    for i in range(len(images)):
        ri = r[i]
        images2.append(images[ri])
        labels2.append(labels[ri])
    return np.asarray(images2), np.asarray(labels2)


def center_samples(images):
    images -= np.mean(images, keepdims=True)
    images /= (np.std(images, keepdims=True) + K.epsilon())


def get_svhn_digit_data():
    image_files = sorted([f for f in os.listdir('input/digits') if f.endswith('.png')])
    init_images=  []
    for f in image_files:
        image = cv2.cvtColor(cv2.imread('input/digits/' + f), cv2.COLOR_BGR2RGB)
        if image.shape[0] > image.shape[1]:
            continue
        start_x = int((image.shape[1] - image.shape[0]) / 2)
        init_images.append(cv2.resize(image[:, start_x:start_x+image.shape[1]], (32, 32)))
    return np.asarray(init_images), np.ones((len(init_images)))
