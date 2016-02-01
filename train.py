#!/usr/bin/env python
from os import path
from scripts.create_dataset import binaryze_dataset, add_salt_and_pepper
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD, Adadelta, Adagrad, Adamax, RMSprop
#from keras.utils.visualize_util import plot

from keras.utils import np_utils
from keras.datasets import mnist

plt.ion()
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (6,4)

np.random.seed(1234)

PATH_SAVE='datasets/mnist/'

#shape='spirals'
#optimizer = SGD(lr=0.5, decay=1e-1, momentum=0.9, nesterov=True)
#optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
optimizer = RMSprop()
#optimizer = Adagrad(lr=1.0, epsilon=1e-06)
#optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
train_size=50000
num_epochs=30
batch_size=100
nb_classes=2

if nb_classes == 2:
    loss='binary_crossentropy'
    loss='mse'
else:
    loss='categorical_crossentropy'

def compute_loss(prob, Y, loss='mse'):
    if loss == 'mse':
        error = np.mean(np.square(prob-Y))
    elif loss == 'binary_crossentropy':
        error = -1
    return error

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

def create_mlp(num_out=10, activation='sigmoid'):
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_out))
    model.add(Activation(activation))
    return model

def reliability_diagram(prob, Y, label=''):
    hist_tot = np.histogram(prob, bins=np.linspace(0,1,11))
    hist_pos = np.histogram(prob[Y == 1], bins=np.linspace(0,1,11))
    plt.plot([0,1],[0,1], 'r--')
    plt.plot(hist_pos[1][:-1]+0.05, np.true_divide(hist_pos[0],hist_tot[0]+1),
             'x-', linewidth=2.0, label=label)

def plot_reliability_diagram(prob_train, Y_train, prob_val, Y_val, epoch,
                             save=True):
    fig = plt.figure('reliability_diagram')
    plt.clf()
    plt.title('Reliability diagram')
    reliability_diagram(prob_train, Y_train, label='train')
    reliability_diagram(prob_val, Y_val, label='val')
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

def imshow_samples(X_train, y_train, X_val, y_val, num_samples=4, save=True):
    fig = plt.figure('samples')
    for j, (data_x, data_y) in enumerate([(X_train, y_train), (X_val, y_val)]):
        for i in range(num_samples):
            plt.subplot(2,num_samples,(j*num_samples+i)+1)
            plt.imshow(np.reshape(data_x[i], (28,28)))
            plt.title(data_y[i])
    plt.show()
    if save:
        plt.savefig('samples.svg')

def plot_accuracy(accuracy_train, accuracy_val, epoch, save=True):
    fig = plt.figure('accuracy')
    plt.clf()
    plt.title("Accuracy")
    plt.plot(range(0,epoch), accuracy_train[:epoch], label='train')
    plt.plot(range(0,epoch), accuracy_val[:epoch], label='val')
    plt.legend(loc='lower right')
    plt.show()
    if save:
        plt.savefig('accuracy_{:03}.svg'.format(epoch))

def plot_error(error_train, error_val, epoch, loss, save=True):
    fig = plt.figure('error')
    plt.clf()
    plt.title(loss)
    plt.plot(range(0,epoch), error_train[:epoch], label='train')
    plt.plot(range(0,epoch), error_val[:epoch], label='val')
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
    model = create_mlp(num_out=1, activation='tanh')
    model.compile(optimizer=optimizer, loss=loss, class_mode='binary')
else:
    model = create_mlp(num_out=nb_classes, activation='softmax')
    model.compile(optimizer=optimizer, loss=loss)
#plot(model, to_file='model.png')

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_val = X_train[train_size:]
y_val = y_train[train_size:]
X_train = X_train[:train_size]
y_train = y_train[:train_size]

X_train, Y_train = preprocess_data(X_train, y_train, nb_classes=nb_classes,
        binarize=True, noise=True, proportion=0.25)
X_val, Y_val = preprocess_data(X_val, y_val, nb_classes=nb_classes,
        binarize=True, noise=True, proportion=0.25)

Y_train_neg = np.copy(Y_train)
Y_train_neg[Y_train_neg==0] = -1
Y_val_neg = np.copy(Y_val)
Y_val_neg[Y_val_neg==0] = -1

imshow_samples(X_train, y_train, X_val, y_val, 5)

error_train  = np.zeros(num_epochs+1)
error_val = np.zeros(num_epochs+1)
accuracy_train = np.zeros(num_epochs+1)
accuracy_val = np.zeros(num_epochs+1)
score = model.evaluate(X_train, Y_train_neg, batch_size=batch_size, show_accuracy=True)
error_train[0] = score[0]
accuracy_train[0] = score[1]
score = model.evaluate(X_val, Y_val_neg, batch_size=batch_size, show_accuracy=True)
error_val[0] = score[0]
accuracy_val[0] = score[1]

ir = IsotonicRegression(out_of_bounds='clip')
prob_train = model.predict(X_train).flatten()
prob_ir = ir.fit_transform(prob_train, Y_train)
Y_ir = Y_train_neg + prob_ir

for epoch in range(1,num_epochs+1):
    hist = model.fit(X_train, Y_ir, nb_epoch=1, batch_size=batch_size,
                        show_accuracy=True)

    prob_train = model.predict(X_train).flatten()
    prob_val = model.predict(X_val).flatten()

    prob_train_ir  = ir.fit_transform(prob_train.flatten(), Y_train)
    prob_val_ir  = ir.predict(prob_val.flatten())
    Y_ir = Y_train_neg + prob_train_ir

    error_train[epoch] = compute_loss(prob_train_ir, Y_train, loss)
    error_val[epoch] = compute_loss(prob_val_ir, Y_val, loss)
    print("EPOCH {}, train error = {}, val error = {}".format(epoch, error_train[epoch], error_val[epoch]))

    accuracy_train[epoch] = compute_accuracy(prob_train_ir, Y_train)
    accuracy_val[epoch] = compute_accuracy(prob_val_ir, Y_val)
    print"Train acc = {}, Val acc = {}".format(accuracy_train[epoch],
                                               accuracy_val[epoch])

    fig = plt.figure('IR')
    plt.clf()
    plt.scatter(prob_train, prob_train_ir)
    plt.show()
    plt.savefig('ir_{:03}.svg'.format(epoch))

    # PLOTS
    plot_reliability_diagram(prob_train_ir, Y_train, prob_val_ir, Y_val, epoch)
    plot_histogram_scores(prob_train_ir, epoch)
    plot_accuracy(accuracy_train, accuracy_val, epoch)
    plot_error(error_train, error_val, epoch, loss)
    plt.pause(0.1)
