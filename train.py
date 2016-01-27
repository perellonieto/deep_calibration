#!/usr/bin/env python
from os import path
from scripts.create_dataset import binaryze_dataset, add_salt_and_pepper
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1234)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD, Adadelta, Adagrad, Adamax
from keras.utils.visualize_util import plot

from keras.utils import np_utils
from keras.datasets import mnist

PATH_SAVE='datasets/mnist/'

#shape='spirals'
#optimizer = SGD(lr=0.5, decay=1e-1, momentum=0.9, nesterov=True)
optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#optimizer = Adagrad(lr=1.0, epsilon=1e-06)
#optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
num_epochs=30
batch_size=100
loss='binary_crossentropy'

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
    model.add(Activation('sigmoid'))
    return model

model = create_mlp(num_out=10)
plot(model, to_file='model.png')
model.compile(optimizer=optimizer, loss=loss)

def reliability_diagram(prob, Y):
    hist_tot = np.histogram(prob, bins=np.linspace(0,1,11))
    hist_pos = np.histogram(prob[Y == 1], bins=np.linspace(0,1,11))
    plt.plot([0,1],[0,1], 'r--')
    plt.plot(hist_pos[1][:-1]+0.05, np.true_divide(hist_pos[0],hist_tot[0]+1),
             'bx-', linewidth=2.0)

def compute_accuracy(scores, labels, threshold=0.5):
    return np.mean((scores >= threshold) == labels)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train /= 255
#X_test /= 255
X_train = binaryze_dataset(X_train)
X_train = add_salt_and_pepper(X_train,proportion=0.25)

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

##X = np.load(path.join(PATH_SAVE, "training_X.npy"))
###X = np.reshape(X, (np.shape(X)[0],1,28,28))
##X = np.reshape(X, (np.shape(X)[0],-1))
##Y = np.load(path.join(PATH_SAVE, "training_Y.npy"))
##
### FIXME : I am reducing the size of the dataset to test the code
##X = X[0:10000]
##Y = Y[0:10000]
##Y = np.in1d(Y,[0,2,4,6,8])
###Y[Y!=5] = 0
###Y[Y==5] = 1

error = np.zeros(num_epochs+1)
accuracy = np.zeros(num_epochs+1)
score = model.evaluate(X_train, Y_train, batch_size=batch_size, show_accuracy=True)
error[0] = score[0]
accuracy[0] = score[1]
for epoch in range(1,num_epochs+1):
    #X_train, Y_train = generate_samples(shape=shape, samples=samples)
    model.fit(X_train, Y_train, nb_epoch=1, batch_size=batch_size,
              show_accuracy=True)

    score = model.evaluate(X_train, Y_train, batch_size=batch_size, show_accuracy=True)
    error[epoch] = score[0]
    print("EPOCH {}, error = {}".format(epoch, error[epoch]))

    prob = model.predict(X_train)
    fig = plt.figure('reliability_diagram')
    plt.clf()
    plt.title('Reliability diagram')
    reliability_diagram(prob, Y_train)
    plt.show()
    plt.savefig('rel_dia_{:03}.svg'.format(epoch))

    fig = plt.figure('histogram_scores')
    plt.clf()
    plt.title('Histogram of scores')
    plt.hist(prob)
    plt.show()
    plt.savefig('hist_scor_{:03}.svg'.format(epoch))

    accuracy[epoch] = score[1]
    fig = plt.figure('accuracy')
    plt.clf()
    plt.title('Accuracy')
    plt.plot(range(0,epoch), accuracy[:epoch])
    plt.show()
    plt.savefig('accuracy_{:03}.svg'.format(epoch))

    fig = plt.figure('error', figsize=(6,4))
    plt.clf()
    plt.title(loss)
    plt.plot(range(0,epoch), error[:epoch])
    plt.ylabel(loss)
    plt.xlabel('epoch')
    plt.show()
    plt.savefig('{}_{:03}.svg'.format(loss, epoch))

    plt.pause(0.1)

fig = plt.figure('error', figsize=(6,4))
plt.clf()
plt.title(loss)
plt.plot(range(0,num_epochs+1), error[0:])
plt.ylabel(loss)
plt.xlabel('epoch')
plt.show()
plt.savefig('{}.svg'.format(loss))

accuracy[epoch] = compute_accuracy(prob, Y_train)
fig = plt.figure('accuracy')
plt.clf()
plt.title("Accuracy")
plt.plot(range(0,epoch), accuracy[:epoch])
plt.show()
plt.savefig('accuracy_{:03}.svg'.format(epoch))
