#!/usr/bin/env python
from os import path
from scripts.create_dataset import binaryze_dataset, add_salt_and_pepper
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (6,4)


np.random.seed(1234)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD, Adadelta, Adagrad, Adamax, RMSprop
#from keras.utils.visualize_util import plot

from keras.utils import np_utils
from keras.datasets import mnist

PATH_SAVE='datasets/mnist/'

#shape='spirals'
#optimizer = SGD(lr=0.5, decay=1e-1, momentum=0.9, nesterov=True)
#optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
optimizer = RMSprop()
#optimizer = Adagrad(lr=1.0, epsilon=1e-06)
#optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
train_size=50
num_epochs=30
batch_size=100
nb_classes=2

if nb_classes == 2:
    loss='binary_crossentropy'
    loss='mse'
else:
    loss='categorical_crossentropy'

def create_cnn():
    model = Sequential()
    model.add(Convolution2D(nb_filter=32, nb_row=2, nb_col=2, border_mode='valid',
                            init='glorot_uniform', input_shape=(1,28,28),
                            activation='relu', dim_ordering='th'))
    model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3,
                            init='glorot_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3,
                            init='glorot_uniform', activation='relu'))
    model.add(Convolution2D(nb_filter=32, nb_row=3, nb_col=3,
                            init='glorot_uniform', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(128, init='glorot_uniform', activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, init='glorot_uniform', activation='sigmoid'))

    return model

def create_mlp(num_out=10):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_out))
    if num_out == 1:
        model.add(Activation('sigmoid'))
    else:
        model.add(Activation('softmax'))
    return model

def reliability_diagram(prob, Y, label=''):
    hist_tot = np.histogram(prob, bins=np.linspace(0,1,11))
    hist_pos = np.histogram(prob[Y == 1], bins=np.linspace(0,1,11))
    plt.plot([0,1],[0,1], 'r--')
    plt.plot(hist_pos[1][:-1]+0.05, np.true_divide(hist_pos[0],hist_tot[0]+1),
             'x-', linewidth=2.0, label=label)

def plot_reliability_diagram(prob_train, Y_train, prob_test, Y_test, epoch,
                             save=True):
    fig = plt.figure('reliability_diagram')
    plt.clf()
    plt.title('Reliability diagram')
    reliability_diagram(prob_train, Y_train, label='train')
    reliability_diagram(prob_test, Y_test, label='test')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
    if save:
        plt.savefig('rel_dia_{:03}.svg'.format(epoch))

def compute_accuracy(scores, labels, threshold=0.5):
    return np.mean((scores >= threshold).flatten() == labels.flatten())

def preprocess_data(X,y,nb_classes=10, binarize=False, noise=False,
                    proportion=0.1):
    X = X.reshape(-1, 784)
    X = X.astype('float32')
    X /= 255.0
    print(X.shape[0], 'samples')
    if binarize:
        X = binaryze_dataset(X, threshold=0.5)
    if noise:
        X = add_salt_and_pepper(X,proportion=proportion)
    if nb_classes == 2:
        Y = np.in1d(y,[0,2,4,6,8]).astype('float64')
    else:
        Y = np_utils.to_categorical(y,nb_classes)
    return X,Y

def imshow_samples(X_train, y_train, X_test, y_test, num_samples=4, save=True):
    fig = plt.figure('samples')
    for j, (data_x, data_y) in enumerate([(X_train, y_train), (X_test, y_test)]):
        for i in range(num_samples):
            plt.subplot(2,num_samples,(j*num_samples+i)+1)
            plt.imshow(np.reshape(data_x[i], (28,28)))
            plt.title(data_y[i])
    plt.show()
    if save:
        plt.savefig('samples.svg')

def plot_accuracy(accuracy_train, accuracy_test, epoch, save=True):
    fig = plt.figure('accuracy')
    plt.clf()
    plt.title("Accuracy")
    plt.plot(range(0,epoch), accuracy_train[:epoch], label='train')
    plt.plot(range(0,epoch), accuracy_test[:epoch], label='test')
    plt.legend(loc='lower right')
    plt.show()
    if save:
        plt.savefig('accuracy_{:03}.svg'.format(epoch))

def plot_error(error_train, error_test, epoch, loss, save=True):
    fig = plt.figure('error')
    plt.clf()
    plt.title(loss)
    plt.plot(range(0,epoch), error_train[:epoch], label='train')
    plt.plot(range(0,epoch), error_test[:epoch], label='test')
    plt.legend()
    plt.ylabel(loss)
    plt.xlabel('epoch')
    plt.show()
    if save:
        plt.savefig('{}_{:03}.svg'.format(loss, epoch))

def plot_histogram_scores(scores, epoch, save=True):
    fig = plt.figure('histogram_scores')
    plt.clf()
    plt.title('Histogram of scores (train)')
    plt.hist(scores)
    plt.show()
    if save:
        plt.savefig('hist_scor_{:03}.svg'.format(epoch))

if nb_classes == 2:
    model = create_mlp(num_out=1)
    model.compile(optimizer=optimizer, loss=loss, class_mode='binary')
else:
    model = create_mlp(num_out=nb_classes)
    model.compile(optimizer=optimizer, loss=loss)
#plot(model, to_file='model.png')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train[:train_size]
y_train = y_train[:train_size]

X_train, Y_train = preprocess_data(X_train, y_train, nb_classes=nb_classes)
#        binarize=True, noise=True, proportion=0.4)
X_test, Y_test = preprocess_data(X_test, y_test, nb_classes=nb_classes)
#        binarize=True, noise=True, proportion=0.4)

imshow_samples(X_train, y_train, X_test, y_test, 5)

error_train  = np.zeros(num_epochs+1)
error_test = np.zeros(num_epochs+1)
accuracy_train = np.zeros(num_epochs+1)
accuracy_test = np.zeros(num_epochs+1)
score = model.evaluate(X_train, Y_train, batch_size=batch_size, show_accuracy=True)
error_train[0] = score[0]
accuracy_train[0] = score[1]
score = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy=True)
error_test[0] = score[0]
accuracy_test[0] = score[1]
for epoch in range(1,num_epochs+1):
    hist = model.fit(X_train, Y_train, nb_epoch=1, batch_size=batch_size,
                        show_accuracy=True)

    error_train[epoch] = hist.history['loss'][0]
    accuracy_train[epoch] = hist.history['acc'][0]

    score_test = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy=True)
    error_test[epoch] = score_test[0]
    accuracy_test[epoch] = score_test[1]
    print("EPOCH {}, train error = {}, test error = {}".format(epoch, error_train[epoch], error_test[epoch]))


    prob_train = model.predict(X_train)
    prob_test = model.predict(X_test)
    print"Train acc = {}, Val acc = {}".format(compute_accuracy(prob_train,
        Y_train), compute_accuracy(prob_test, Y_test))

    # PLOTS
    plot_reliability_diagram(prob_train, Y_train, prob_test, Y_test, epoch)
    plot_histogram_scores(prob_train, epoch)
    plot_accuracy(accuracy_train, accuracy_test, epoch)
    plot_error(error_train, error_test, epoch, loss)
    plt.pause(0.1)
