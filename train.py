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

plt.ion()
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = (5,3.5)

np.random.seed(1234)

PATH_SAVE='datasets/mnist/'
binarize=True
add_noise=True

#shape='spirals'
#optimizer = SGD(lr=0.5, decay=1e-1, momentum=0.9, nesterov=True)
#optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
optimizer = RMSprop()
#optimizer = Adagrad(lr=1.0, epsilon=1e-06)
#optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
train_size=50000
num_epochs=30
batch_size=50000
inner_batch_size=100
nb_classes=2
noise_proportion=0.25
score_lin=np.linspace(0,1,100)
minibatch_method='lineal' # 'random', 'lineal'
output_activation= 'sigmoid' # 'isotonic_regression' # sigmoid

if nb_classes == 2:
    loss='binary_crossentropy'
else:
    loss='categorical_crossentropy'

diary = Diary(name='experiment', path='results')
diary.add_notebook('hyperparameters')
diary.add_entry('hyperparameters', ['train_size', train_size,
    'epoch', num_epochs, 'batch_size', batch_size, 'classes', nb_classes,
    'inner_batch_size', inner_batch_size,
    'minibatch_method', minibatch_method,
    'output_activation', output_activation, 'loss', loss,
    'optimizer', optimizer.get_config()['name'], 'noise', noise_proportion])
diary.add_notebook('training')
diary.add_notebook('validation')

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
        error = np.mean(np.square(prob-Y))
    elif loss == 'binary_crossentropy':
        error = np.mean(-np.multiply(Y,np.log2(prob)) - np.multiply((1-Y),
                np.log2(1-prob)))
    else:
        print('compute_loss loss = {} not implemented returning -1'.format(loss))
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

def reliability_diagram(prob, Y, marker='--', label=''):
    # TODO modify the centers of the bins by their centroids
    hist_tot = np.histogram(prob, bins=np.linspace(0,1,11))
    hist_pos = np.histogram(prob[Y == 1], bins=np.linspace(0,1,11))
    plt.plot([0,1],[0,1], 'r--')
    plt.plot(hist_pos[1][:-1]+0.05, np.true_divide(hist_pos[0],hist_tot[0]+1),
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

def imshow_samples(X_train, y_train, X_val, y_val, num_samples=4):
    fig = plt.figure('samples')
    for j, (data_x, data_y) in enumerate([(X_train, y_train), (X_val, y_val)]):
        for i in range(num_samples):
            plt.subplot(2,num_samples,(j*num_samples+i)+1)
            plt.imshow(np.reshape(data_x[i], (28,28)))
            plt.title(data_y[i])
    plt.draw()

def plot_accuracy(accuracy_train, accuracy_val, epoch):
    fig = plt.figure('accuracy')
    plt.clf()
    plt.title("Accuracy")
    plt.plot(range(0,epoch), accuracy_train[:epoch], 'x-', label='train')
    plt.plot(range(0,epoch), accuracy_val[:epoch], '+-', label='val')
    plt.legend(loc='lower right')
    plt.draw()

def plot_error(error_train, error_val, epoch, loss):
    fig = plt.figure('error')
    plt.clf()
    plt.title(loss)
    plt.plot(range(0,epoch), error_train[:epoch], 'x-', label='train')
    plt.plot(range(0,epoch), error_val[:epoch], '+-', label='val')
    plt.legend()
    plt.ylabel(loss)
    plt.xlabel('epoch')
    plt.draw()

def plot_histogram_scores(scores, epoch):
    fig = plt.figure('histogram_scores')
    plt.clf()
    plt.title('Histogram of scores (train)')
    plt.hist(scores, bins=np.linspace(0,1,11))
    plt.draw()

def isotonic_gradients(ir, scores, delta=0.01):
#    lower_value = ir.predict(np.clip(scores-delta, a_min=0, a_max=1))
#    upper_value = ir.predict(np.clip(scores+delta, a_min=0, a_max=1))
#    # FIXME the denominator is wrong on clipped samples
#    return (upper_value-lower_value)/delta*2
    return -np.ones(np.shape(scores))

if nb_classes == 2:
    model = create_mlp(num_out=1, activation='sigmoid')
    #model = create_mlp(num_out=1, activation='linear')
    model.compile(optimizer=optimizer, loss=loss, class_mode='binary')
else:
    model = create_mlp(num_out=nb_classes, activation='softmax')
    model.compile(optimizer=optimizer, loss=loss)

if keras_plot_available:
    plot(model, to_file='model.png')

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

print('Showing data samples')
imshow_samples(X_train, y_train, X_val, y_val, 5)
diary.save_figure(plt, filename='samples', extension='svg')

print('Creating error and accuracy vectors')
error_train  = np.zeros(num_epochs+1)
error_val = np.zeros(num_epochs+1)
accuracy_train = np.zeros(num_epochs+1)
accuracy_val = np.zeros(num_epochs+1)

# This are the same points 4 and 5 as in the training loop
# 4. Calibrate the network with isotonic regression in the full training
#   a. Get the new scores from the model
ir = IsotonicRegression(increasing=True, out_of_bounds='clip',
                        y_min=0.000001, y_max=0.9999999)
print('Model predict training scores')
score_train = model.predict(X_train).flatten()
if output_activation == 'isotonic_regression':
    #   b. Calibrate the scores
    print('Learning Isotonic Regression from TRAINING set')
    ir.fit(score_train, Y_train)
    # 5. Evaluate the performance with the calibrated probabilities
    #   a. Evaluation on training set
    prob_train = ir.predict(score_train)
else:
    prob_train = score_train
error_train[0] = compute_loss(prob_train, Y_train, loss)
accuracy_train[0] = compute_accuracy(prob_train, Y_train)
#   b. Evaluation on validation set
print('Model predict validation scores')
score_val = model.predict(X_val).flatten()
if output_activation == 'isotonic_regression':
    print('IR predict validation probabilities')
    prob_val  = ir.predict(score_val.flatten())
else:
    prob_val = score_val
error_val[0] = compute_loss(prob_val, Y_val, loss)
accuracy_val[0] = compute_accuracy(prob_val, Y_val)

# SHOW INITIAL PERFORMANCE
print(("train error = {}, val error = {}\n"
       "train acc = {}, val acc = {}").format(
                    error_train[0], error_val[0],
                    accuracy_train[0], accuracy_val[0]))
diary.add_entry('training', [error_train[0], accuracy_train[0]])
diary.add_entry('validation', [error_val[0], accuracy_val[0]])

# FIXME change epoch by minibatch
num_minibatches = np.ceil(np.true_divide(train_size,batch_size)).astype('int')
for epoch in range(1,num_epochs+1):
    partial_acc_train = 0
    partial_acc_val = 0
    partial_err_train = 0
    partial_err_val = 0
    for iteration in range(num_minibatches):
        # Given the Calibrated probabilities
        # 1. Choose the next minibatch
        print('EPOCH {}'.format(epoch))
        minibatch_id = get_minibatch_id(train_size, batch_size,
                                         method=minibatch_method,
                                         iteration=iteration)
        X_train_mb = np.copy(X_train[minibatch_id])
        Y_train_mb = np.copy(Y_train[minibatch_id])

        # 2. Compute the new values for the labels on this minibatch
        #   a. Predict the scores using the network
        print('\tPREDICTING TRAINING')
        score_train_mb = model.predict(X_train_mb).flatten()
        if output_activation == 'isotonic_regression':
            #   b. Predict the probabilities using IR
            print('\tPREDICTING TRAINING')
            prob_train_mb = ir.predict(score_train_mb.flatten())
            #   c. Compute the gradients of IR
            g_prob_train_mb = isotonic_gradients(ir, prob_train_mb)
            #   c. Compute new values for the labels
            Y_train_mb_new = prob_train_mb + \
                             np.divide(np.multiply(prob_train_mb-Y_train_mb,
                                                   g_prob_train_mb),
                                       np.multiply(prob_train_mb,1-prob_train_mb))
        else:
            prob_train_mb = score_train_mb
            Y_train_mb_new = Y_train_mb

        # 3. Train the network on this minibatch
        print('\tTRAINING MODEL')
        model.fit(X_train_mb, Y_train_mb_new, nb_epoch=1,
                batch_size=inner_batch_size, show_accuracy=True)

        # 4. Calibrate the network with isotonic regression in the full training
        #   a. Get the new scores from the model
        print('\tModel predict training scores')
        score_train = model.predict(X_train).flatten()

        if output_activation == 'isotonic_regression':
            #   b. Calibrate the scores
            print('\tLearning Isotonic Regression from TRAINING set')
            ir.fit(score_train, Y_train)

            # 5. Evaluate the performance with the calibrated probabilities
            #   a. Evaluation on TRAINING set
            prob_train = ir.predict(score_train)
        else:
            prob_train = score_train

        partial_acc_train += compute_accuracy(prob_train, Y_train)
        partial_err_train += compute_loss(prob_train, Y_train, loss)
        #   b. Evaluation on VALIDATION set
        print('\tModel predict validation scores')
        score_val = model.predict(X_val).flatten()
        if output_activation == 'isotonic_regression':
            print('\tIR predict validation probabilities')
            prob_val  = ir.predict(score_val.flatten())
        else:
            prob_val = score_val

        partial_acc_val += compute_accuracy(prob_val, Y_val)
        partial_err_val += compute_loss(prob_val, Y_val, loss)

    error_train[epoch] = partial_err_train/num_minibatches
    accuracy_train[epoch] = partial_acc_train/num_minibatches
    error_val[epoch] = partial_err_val/num_minibatches
    accuracy_val[epoch] = partial_acc_val/num_minibatches
    # SHOW PERFORMANCE ON MINIBATCH
    print(("\ttrain error = {}, val error = {}\n"
           "\ttrain acc = {}, val acc = {}").format(
                        error_train[epoch], error_val[epoch],
                        accuracy_train[epoch], accuracy_val[epoch]))

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
