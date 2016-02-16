import os
import six.moves.cPickle as pickle
import gzip

import numpy

import theano
import theano.tensor as T

from keras.utils import np_utils

from scripts.create_dataset import binaryze_dataset, add_salt_and_pepper

def preprocess_data(X,y,nb_classes=10, binarize=False, noise_prop=0.49):
    X_new = X.reshape(-1, 784)
    X_new = X_new.astype('float32')
    print(X_new.shape[0], 'samples')
    if binarize:
        X_new = binaryze_dataset(X_new, threshold=0.5)
    if noise_prop != None:
        print("Adding salt and pepper ({})".format(noise_prop))
        X_new = add_salt_and_pepper(X_new,proportion=noise_prop)
    if nb_classes == 2:
        Y = numpy.in1d(y,[0,2,4,6,8]).astype('float64')
    else:
        Y = y
        print(Y.shape)
    return X_new,Y

def load_data(dataset, nb_classes=10, binarize=False, noise_prop=None):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "datasets",
            "mnist",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_x, data_y, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')


    test_set_x, test_set_y = test_set
    test_set_x, test_set_y = preprocess_data(test_set_x, test_set_y,
            nb_classes=nb_classes, binarize=binarize, noise_prop=noise_prop)
    valid_set_x, valid_set_y = valid_set
    valid_set_x, valid_set_y = preprocess_data(valid_set_x, valid_set_y,
            nb_classes=nb_classes, binarize=binarize, noise_prop=noise_prop)
    train_set_x, train_set_y = train_set
    train_set_x, train_set_y = preprocess_data(train_set_x, train_set_y,
            nb_classes=nb_classes, binarize=binarize, noise_prop=noise_prop)

    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)
    valid_set_x, valid_set_y = shared_dataset(valid_set_x, valid_set_y)
    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


