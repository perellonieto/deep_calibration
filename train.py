#!/usr/bin/env python
import os
import sys
import timeit

from os import path
from scripts.create_dataset import binaryze_dataset, add_salt_and_pepper
import numpy as np
import numpy
from sklearn.isotonic import IsotonicRegression
from data import load_data

import theano
import theano.tensor as T
from mlp import MLP

# FIXME the code is not working with TensorFlow, the error loss is not computed
# correctly
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD, Adadelta, Adagrad, Adamax, RMSprop

try:
    from keras.utils.visualize_util import plot
    keras_plot_available = True
except ImportError:
    keras_plot_available = False
except RuntimeError:
    keras_plot_available = False

from keras.utils import np_utils
from keras.datasets import mnist

from diary import Diary

import matplotlib.pyplot as plt
plt.ion()
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (5,3.5)

np.random.seed(1234)
_EPSILON=10e-8

PATH_SAVE='datasets/mnist/'
binarize=False
add_noise=False

#optimizer = SGD(lr=0.5, decay=1e-1, momentum=0.9, nesterov=False)
optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
#optimizer = RMSprop()
#optimizer = Adagrad(lr=1.0, epsilon=1e-06)
#optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
train_size=50000
num_epochs=30
batch_size=5000
inner_batch_size=5000
nb_classes=2
noise_proportion=0.25
score_lin=np.linspace(0,1,100)
minibatch_method='lineal' # 'random', 'lineal'
n_hidden=[25, 25]
output_activation= 'sigmoid' # 'isotonic_regression' # sigmoid

if nb_classes == 2:
    loss='binary_crossentropy'
else:
    loss='categorical_crossentropy'

diary = Diary(name='experiment', path='results')
diary.add_notebook('hyperparameters')
diary.add_entry('hyperparameters', ['train_size', train_size])
diary.add_entry('hyperparameters', ['num_classes', nb_classes])
diary.add_entry('hyperparameters', ['batch_size', batch_size])
diary.add_entry('hyperparameters', ['inner_batch_size', inner_batch_size])
diary.add_entry('hyperparameters', ['minibatch_method', minibatch_method])
diary.add_entry('hyperparameters', ['output_activation', output_activation])
diary.add_entry('hyperparameters', ['loss', loss])
diary.add_entry('hyperparameters', ['optimizer', optimizer.get_config()['name']])
for key, value in optimizer.get_config().iteritems():
    diary.add_entry('hyperparameters', [key, value])
diary.add_entry('hyperparameters', ['binarize', binarize])
diary.add_entry('hyperparameters', ['add_noise', add_noise])
diary.add_entry('hyperparameters', ['noise', noise_proportion])
diary.add_notebook('training')
diary.add_notebook('validation')

def get_mnist_data(train_size, binarize, add_noise, noise_proportion,
        test=False):
    print('Loading MNIST dataset')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print('Splitting data into training and validation')
    X_val = np.copy(X_train[train_size:])
    y_val = np.copy(y_train[train_size:])
    X_train = np.copy(X_train[:train_size])
    y_train = np.copy(y_train[:train_size])
    print('Training shape = {}, Validation shape = {}, Test shape = {}'.format(
        X_train.shape, X_val.shape, X_test.shape))

    print('Preprocessing data: classes = {}, binarize = {}, add noise = {}'.format(
           nb_classes, binarize, add_noise))
    X_train, Y_train = preprocess_data(X_train, y_train, nb_classes=nb_classes,
            binarize=binarize, noise=add_noise, proportion=noise_proportion)
    X_val, Y_val = preprocess_data(X_val, y_val, nb_classes=nb_classes,
            binarize=binarize, noise=add_noise, proportion=noise_proportion)
    if test:
        X_test, Y_test = preprocess_data(X_test, y_test, nb_classes=nb_classes,
            binarize=binarize, noise=add_noise, proportion=noise_proportion)
        return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    else:
        return (X_train, Y_train), (X_val, Y_val)

def get_minibatch_id(total_size, batch_size, method='random', iteration=0):
    if method == 'random':
        minibatch_id = np.random.choice(total_size, batch_size)
    elif 'lineal':
        minibatch_id = np.mod(range(batch_size*iteration,
                                    batch_size*(iteration+1)),
                              total_size)
    return minibatch_id

def compute_loss(prob, Y, loss='mse'):
    if loss == 'mse':
        error = np.mean(np.square(prob - Y))
    elif loss == 'binary_crossentropy':
        prob_clip = prob.clip(_EPSILON, 1.0 - _EPSILON)
        error = -np.mean(np.multiply(Y, np.log(prob_clip)) +
                         np.multiply((1.0 - Y), np.log(1.0 - prob_clip)))
    else:
        print('compute_loss loss = {} not implemented (-1)'.format(loss))
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

def create_model(classes, optimizer, loss):
    if nb_classes == 2:
        model = create_mlp(num_out=1, activation='sigmoid')
        model.compile(optimizer=optimizer, loss=loss, class_mode='binary')
    else:
        model = create_mlp(num_out=nb_classes, activation='softmax')
        model.compile(optimizer=optimizer, loss=loss)
    return model

def reliability_diagram(prob, Y, marker='--', label=''):
    # TODO modify the centers of the bins by their centroids
    hist_tot = np.histogram(prob, bins=np.linspace(0,1,11))
    hist_pos = np.histogram(prob[Y == 1], bins=np.linspace(0,1,11))
    plt.plot([0,1],[0,1], 'r--')
    centroids = [np.mean(np.append(prob[(prob > hist_tot[1][i]) * (prob <
        hist_tot[1][i+1])], hist_tot[1][i]+0.05)) for i in range(len(hist_tot[1])-1)]
    plt.plot(centroids, np.true_divide(hist_pos[0]+1,hist_tot[0]+2),
             marker, linewidth=2.0, label=label)

def plot_reliability_diagram(prob_train, Y_train, prob_val, Y_val, epoch,
                             score_lin=None, prob_lin=None):
    fig = plt.figure('reliability_diagram')
    plt.clf()
    plt.title('Reliability diagram')
    reliability_diagram(prob_train, Y_train, marker='x-', label='train.')
    reliability_diagram(prob_val, Y_val, marker='+-', label='val.')
    if score_lin != None and prob_lin != None:
        plt.plot(score_lin, prob_lin, label='cal.')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.draw()

def compute_accuracy(scores, labels, threshold=0.5):
    return np.mean((scores >= threshold).flatten() == labels.flatten())

def preprocess_data(X,y,nb_classes=10, binarize=False, noise=False,
                    proportion=0.1):
    X_new = X.reshape(-1, 784)
    X_new = X_new.astype('float32')
    X_new /= 255.0
    print(X_new.shape[0], 'samples')
    if binarize:
        X_new = binaryze_dataset(X_new, threshold=0.5)
    if noise:
        X_new = add_salt_and_pepper(X_new,proportion=proportion)
    if nb_classes == 2:
        Y = np.in1d(y,[0,2,4,6,8]).astype('float64')
    else:
        Y = np_utils.to_categorical(y,nb_classes)
    return X_new,Y

def imshow_samples(X_train, Y_train, X_val, Y_val, num_samples=4,
        labels=[0,1,2,3,4,5,6,7,8,9]):
    fig = plt.figure('samples')
    for j, (data_x, data_y) in enumerate([(X_train, Y_train), (X_val, Y_val)]):
        for i in range(num_samples):
            plt.subplot(2,num_samples,(j*num_samples+i)+1)
            plt.imshow(np.reshape(data_x[i], (28,28)))
            plt.title(labels[data_y[i].eval()])
    plt.draw()

def plot_accuracy(accuracy_train, accuracy_val, epoch):
    fig = plt.figure('accuracy')
    plt.clf()
    plt.title("Accuracy")
    plt.plot(range(1,epoch+1), accuracy_train[1:epoch+1], 'x-', label='train')
    plt.plot(range(1,epoch+1), accuracy_val[1:epoch+1], '+-', label='val')
    plt.legend(loc='lower right')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.draw()

def plot_error(error_train, error_val, epoch, loss):
    fig = plt.figure('error')
    plt.clf()
    plt.title(loss)
    plt.plot(range(1,epoch+1), error_train[1:epoch+1], 'x-', label='train')
    plt.plot(range(1,epoch+1), error_val[1:epoch+1], '+-', label='val')
    plt.legend()
    plt.ylabel(loss)
    plt.xlabel('epoch')
    plt.grid(True)
    plt.draw()

def plot_histogram_scores(scores_train, scores_val,  epoch):
    fig = plt.figure('histogram_scores')
    plt.clf()
    plt.title('Histogram of scores (train)')
    plt.hist([scores_train, scores_val], bins=np.linspace(0,1,11))
    plt.grid(True)
    plt.draw()

def isotonic_gradients(ir, scores, delta=0.01):
#    lower_value = ir.predict(np.clip(scores-delta, a_min=0, a_max=1))
#    upper_value = ir.predict(np.clip(scores+delta, a_min=0, a_max=1))
#    # FIXME the denominator is wrong on clipped samples
#    return (upper_value-lower_value)/delta*2
    return -np.ones(np.shape(scores))


def main_old():
    model = create_model(nb_classes, optimizer, loss)

    if keras_plot_available:
        plot(model, to_file='model.png')

    (X_train, Y_train), (X_val, Y_val) = get_mnist_data(
            train_size, binarize, add_noise, noise_proportion, test=False)

    print('Showing data samples')
    imshow_samples(X_train, Y_train, X_val, Y_val, 5)
    diary.save_figure(plt, filename='samples', extension='svg')

    print('Creating error and accuracy vectors')
    error_train  = np.zeros(num_epochs+1)
    error_val = np.zeros(num_epochs+1)
    accuracy_train = np.zeros(num_epochs+1)
    accuracy_val = np.zeros(num_epochs+1)

    print('Model predict training scores')
    score_train = model.predict(X_train).flatten()
    if output_activation == 'isotonic_regression':
        # 4. Calibrate the network with isotonic regression in the full training
        ir = IsotonicRegression(increasing=True, out_of_bounds='clip',
                                y_min=_EPSILON, y_max=(1-_EPSILON))
        #   b. Calibrate the scores
        print('Learning Isotonic Regression from TRAINING set')
        ir.fit(score_train, Y_train)

    # 5. Evaluate the performance with probabilities
    #   b. Evaluation on validation set
    print('Model predict validation scores')
    score_val = model.predict(X_val).flatten()
    if output_activation == 'isotonic_regression':
        prob_train = ir.predict(score_train)
        print('IR predict validation probabilities')
        prob_val  = ir.predict(score_val)
    else:
        prob_train = score_train
        prob_val = score_val

    error_train[0] = compute_loss(prob_train, Y_train, loss)
    accuracy_train[0] = compute_accuracy(prob_train, Y_train)
    error_val[0] = compute_loss(prob_val, Y_val, loss)
    accuracy_val[0] = compute_accuracy(prob_val, Y_val)

    # SHOW INITIAL PERFORMANCE
    print(("train:  error = {}, acc = {}\n"
           "valid:  error = {}, acc = {}").format(
                        error_train[0], accuracy_train[0],
                        error_val[0], accuracy_val[0]))

    diary.add_entry('training', [error_train[0], accuracy_train[0]])
    diary.add_entry('validation', [error_val[0], accuracy_val[0]])

    num_minibatches = np.ceil(np.true_divide(train_size,batch_size)).astype('int')
    for epoch in range(1,num_epochs+1):
        for iteration in range(num_minibatches):
            # Given that the probabilities are calibrated
            # 1. Choose the next minibatch
            print('EPOCH {}'.format(epoch))
            minibatch_id = get_minibatch_id(train_size, batch_size,
                                             method=minibatch_method,
                                             iteration=iteration)
            X_train_mb = X_train[minibatch_id]
            Y_train_mb = Y_train[minibatch_id]

            if output_activation == 'isotonic_regression':
                # 2. Compute the new values for the labels on this minibatch
                #   a. Predict the scores using the network
                print('\tMODEL PREDICTING TRAINING SCORES')
                score_train_mb = model.predict(X_train_mb).flatten()
                #   b. Predict the probabilities using IR
                print('\tIR PREDICTING TRAINING PROBABILITIES')
                prob_train_mb = ir.predict(score_train_mb.flatten())
                #   c. Compute the gradients of IR
                g_prob_train_mb = isotonic_gradients(ir, prob_train_mb)
                #   c. Compute new values for the labels
                #Y_train_mb_new = prob_train_mb + Y_train_mb
                Y_train_mb_new = prob_train_mb + \
                                 np.divide(np.multiply(prob_train_mb - Y_train_mb,
                                                       g_prob_train_mb),
                                           np.multiply(prob_train_mb,
                                                       1 - prob_train_mb))
            else:
                Y_train_mb_new = Y_train_mb

            # 3. Train the network on this minibatch
            #    Be advised that the errors shown by Keras on the training
            #    set are really for this minibatch.
            print('\tTRAINING MODEL')
            model.fit(X_train_mb, Y_train_mb_new, nb_epoch=1,
                    batch_size=inner_batch_size, show_accuracy=True, verbose=1,
                    validation_data=(X_val,Y_val))

            if output_activation == 'isotonic_regression':
                # 4. Calibrate the network with isotonic regression in the full training
                #   a. Get the new scores from the model
                print('\tModel predict training scores')
                score_train = model.predict(X_train).flatten()
                #   b. Calibrate the scores
                print('\tLearning Isotonic Regression from TRAINING set')
                ir.fit(score_train, Y_train)

        # Evaluate epoch on the full training and validation set
        # 5. Evaluate the performance with the calibrated probabilities
        print('\tModel predict training scores')
        score_train = model.predict(X_train).flatten()
        print('\tModel predict validation scores')
        score_val = model.predict(X_val).flatten()
        if output_activation == 'isotonic_regression':
            #   a. Evaluation on TRAINING set
            print('\tIR predict training probabilities')
            prob_train = ir.predict(score_train.flatten())
            #   b. Evaluation on VALIDATION set
            print('\tIR predict validation probabilities')
            prob_val  = ir.predict(score_val.flatten())
        else:
            prob_train = score_train
            prob_val = score_val

        error_train[epoch] = compute_loss(prob_train, Y_train, loss)
        accuracy_train[epoch] = compute_accuracy(prob_train, Y_train)
        error_val[epoch] = compute_loss(prob_val, Y_val, loss)
        accuracy_val[epoch] = compute_accuracy(prob_val, Y_val)
        # SHOW PERFORMANCE ON MINIBATCH
        print(("\ttrain:  error = {}, acc = {}\n"
               "\tvalid:  error = {}, acc = {}").format(
                            error_train[epoch], accuracy_train[epoch],
                            error_val[epoch], accuracy_val[epoch]))

        # SAVE PERFORMANCE ON epoch
        diary.add_entry('training', [error_train[epoch], accuracy_train[epoch]])
        diary.add_entry('validation', [error_val[epoch], accuracy_val[epoch]])

        # PLOTS
        print('\tUpdating all plots')
        if output_activation == 'isotonic_regression':
            prob_lin = ir.predict(score_lin)
            plot_reliability_diagram(prob_train, Y_train, prob_val, Y_val, epoch,
                                     score_lin=score_lin, prob_lin=prob_lin)
        else:
            plot_reliability_diagram(prob_train, Y_train, prob_val, Y_val, epoch)
        diary.save_figure(plt, filename='reliability_diagram', extension='svg')
        plot_histogram_scores(prob_train, epoch)
        diary.save_figure(plt, filename='histogram_scores', extension='svg')
        plot_accuracy(accuracy_train, accuracy_val, epoch)
        diary.save_figure(plt, filename='accuracy', extension='svg')
        plot_error(error_train, error_val, epoch, loss)
        diary.save_figure(plt, filename='error', extension='svg')
        plt.pause(0.0001)

def main(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=[500, 500]):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """
    if add_noise==True:
        datasets = load_data(dataset, nb_classes=nb_classes, binarize=binarize,
                             noise_prop=noise_proportion)
    else:
        datasets = load_data(dataset, nb_classes=nb_classes, binarize=binarize)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print('Showing data samples')
    if nb_classes == 2:
        labels = ['odd', 'even']
    else:
        labels = [0,1,2,3,4,5,6,7,8,9]
    imshow_samples(train_set_x.get_value(), train_set_y,
            valid_set_x.get_value(), valid_set_y, num_samples=4, labels=labels)
    plt.pause(0.0001)
    diary.save_figure(plt, filename='samples', extension='svg')

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=nb_classes
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    training_error_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_loss_model = theano.function(
        inputs=[index],
        outputs=classifier.loss(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validation_loss_model = theano.function(
        inputs=[index],
        outputs=classifier.loss(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    training_loss_model = theano.function(
        inputs=[index],
        outputs=classifier.loss(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_accuracy_model = theano.function(
        inputs=[index],
        outputs=classifier.accuracy(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validation_accuracy_model = theano.function(
        inputs=[index],
        outputs=classifier.accuracy(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    training_accuracy_model = theano.function(
        inputs=[index],
        outputs=classifier.accuracy(y),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
            y: train_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compiling a Theano function that computes the predictions on the
    # training data
    training_predictions_model = theano.function(
        inputs=[index],
        outputs=classifier.predictions(),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
        }
    )

    validation_predictions_model = theano.function(
        inputs=[index],
        outputs=classifier.predictions(),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
        }
    )

    # compiling a Theano function that computes the predictions on the
    # training data
    training_scores_model = theano.function(
        inputs=[index],
        outputs=classifier.scores(),
        givens={
            x: train_set_x[index * batch_size:(index + 1) * batch_size],
        }
    )

    validation_scores_model = theano.function(
        inputs=[index],
        outputs=classifier.scores(),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    error_tra = np.zeros(n_epochs+1)
    error_val = np.zeros(n_epochs+1)
    accuracy_tra = np.zeros(n_epochs+1)
    accuracy_val = np.zeros(n_epochs+1)

    epoch = 0
    # Error in accuracy
    training_loss = [training_loss_model(i) for i
                         in range(n_train_batches)]
    validation_loss = [validation_loss_model(i) for i
                         in range(n_valid_batches)]
    error_tra[epoch] = numpy.mean(training_loss)
    error_val[epoch] = numpy.mean(validation_loss)
    training_acc = [training_accuracy_model(i) for i
                         in range(n_train_batches)]
    validation_acc = [validation_accuracy_model(i) for i
                         in range(n_valid_batches)]
    accuracy_tra[epoch] = numpy.mean(training_acc)
    accuracy_val[epoch] = numpy.mean(validation_acc)
    diary.add_entry('training', [error_tra[epoch], accuracy_tra[epoch]])
    diary.add_entry('validation', [error_val[epoch], accuracy_val[epoch]])

    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

        # Error in accuracy
        training_loss = [training_loss_model(i) for i
                             in range(n_train_batches)]
        validation_loss = [validation_loss_model(i) for i
                             in range(n_valid_batches)]
        error_tra[epoch] = numpy.mean(training_loss)
        error_val[epoch] = numpy.mean(validation_loss)
        training_acc = [training_accuracy_model(i) for i
                             in range(n_train_batches)]
        validation_acc = [validation_accuracy_model(i) for i
                             in range(n_valid_batches)]
        accuracy_tra[epoch] = numpy.mean(training_acc)
        accuracy_val[epoch] = numpy.mean(validation_acc)
        diary.add_entry('training', [error_tra[epoch], accuracy_tra[epoch]])
        diary.add_entry('validation', [error_val[epoch], accuracy_val[epoch]])

        plot_error(error_tra, error_val, epoch, 'loss')
        diary.save_figure(plt, filename='error', extension='svg')
        plot_accuracy(accuracy_tra, accuracy_val, epoch)
        diary.save_figure(plt, filename='accuracy', extension='svg')
        if nb_classes == 2:
            prob_train = np.asarray([training_scores_model(i) for i
                             in range(n_train_batches)]).reshape(-1,nb_classes)
            prob_val = np.asarray([validation_scores_model(i) for i
                             in range(n_valid_batches)]).reshape(-1,nb_classes)
            plot_reliability_diagram(prob_train[:,1], train_set_y.eval(),
                                     prob_val[:,1], valid_set_y.eval(), epoch)
            diary.save_figure(plt, filename='reliability_diagram', extension='svg')
            plot_histogram_scores(prob_train[:,1], prob_val[:,1], epoch=epoch)
            diary.save_figure(plt, filename='histogram_scores', extension='svg')
        #from IPython import embed
        #embed()
        plt.pause(0.0001)

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    #print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

if __name__ == '__main__':
    status = main(n_epochs=num_epochs, n_hidden=n_hidden)
    sys.exit(status)

