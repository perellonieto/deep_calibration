#!/usr/bin/env python
from os import path
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
plt.rcParams['image.cmap'] = 'gray'


np.random.seed(1234)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adadelta, Adagrad, Adamax

PATH_SAVE='datasets/mnist/'

#shape='spirals'
#optimizer = SGD(lr=0.5, decay=1e-1, momentum=0.9, nesterov=True)
optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#optimizer = Adagrad(lr=1.0, epsilon=1e-06)
#optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
num_epochs=30
batch_size=100
loss='binary_crossentropy'

# Model
model = Sequential()
model.add(Dense(100, input_dim=28*28, init='glorot_uniform', activation='tanh'))
model.add(Dense(50, init='glorot_uniform', activation='tanh'))
model.add(Dense(8, input_dim=2, init='glorot_uniform', activation='tanh'))
#model.add(Dense(12, init='uniform', activation='tanh'))
model.add(Dense(1, init='glorot_uniform', activation='sigmoid'))
model.compile(optimizer=optimizer, loss=loss)

def reliability_diagram(prob, Y):
    hist_tot = np.histogram(prob, bins=np.linspace(0,1,11))
    hist_pos = np.histogram(prob[Y == 1], bins=np.linspace(0,1,11))
    plt.plot([0,1],[0,1], 'r--')
    plt.plot(hist_pos[1][:-1]+0.05, np.true_divide(hist_pos[0],hist_tot[0]+1),
             'bx-', linewidth=2.0)
    plt.title('reliability map')

X = np.load(path.join(PATH_SAVE, "training_X.npy"))
X = np.reshape(X, (np.shape(X)[0], -1))
Y = np.load(path.join(PATH_SAVE, "training_Y.npy"))

# FIXME : I am reducing the size of the dataset to test the code
X = X[0:10000]
Y = Y[0:10000]
Y[Y!=5] = 0
Y[Y==5] = 1

error = np.zeros(num_epochs+1)
error[0] = model.evaluate(X, Y, batch_size=batch_size)
for epoch in range(1,num_epochs+1):
    #X, Y = generate_samples(shape=shape, samples=samples)
    model.fit(X, Y, nb_epoch=1, batch_size=batch_size)
    error[epoch] = model.evaluate(X, Y, batch_size=batch_size)
    print("EPOCH {}, error = {}".format(epoch, error[epoch]))

    prob = model.predict(X)
    fig = plt.figure('reliability_diagram')
    plt.clf()
    reliability_diagram(prob, Y)
    plt.show()
    plt.savefig('rel_dia_{:03}.svg'.format(epoch))

    fig = plt.figure('histogram_scores')
    plt.clf()
    plt.hist(prob)
    plt.show()
    plt.savefig('hist_scor_{:03}.svg'.format(epoch))

    plt.pause(0.1)

fig = plt.figure('error', figsize=(6,4))
plt.clf()
plt.plot(range(0,num_epochs+1), error[0:])
plt.ylabel(loss)
plt.xlabel('epoch')
plt.show()
plt.savefig('{}.svg'.format(loss))
