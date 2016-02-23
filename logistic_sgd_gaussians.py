__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy
numpy.random.seed(42)

import theano
import theano.tensor as T

from presentationtier import PresentationTier

from data import generate_gaussian_data
from diary import Diary

n_epochs=50
batch_size=100
learning_rate=0.1

class LogisticRegression(object):
    """Binary Logistic Regression Class

    The logistic regression is fully described by a weight vector :math:`w`
    and bias :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        #self.W = theano.shared(value=numpy.zeros((n_in, n_out),
        #                                         dtype=theano.config.floatX),
        #                        name='W', borrow=True)
        self.w = theano.shared(numpy.random.randn(n_in), name="w")
        # initialize the baises b as a vector of n_out 0s
        #self.b = theano.shared(value=numpy.zeros((n_out,),
        #                                         dtype=theano.config.floatX),
        #                       name='b', borrow=True)
        self.b = theano.shared(0., name="b")

        # compute vector of class-membership probabilities in symbolic form
        #self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.p_y_given_x = 1 / (1 + T.exp(-T.dot(input, self.w) - self.b))

        # compute prediction as class whose probability is maximal in
        # symbolic form
        #self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.y_pred = self.p_y_given_x > 0.5

        # parameters of the model
        self.params = [self.w, self.b]

    def negative_log_likelihood(self, y):
        return T.mean(-y*T.log(self.p_y_given_x) -(1-y)*T.log(1-self.p_y_given_x))

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def scores(self):
        return self.p_y_given_x

    def accuracy(self, y):
        """Return a float representing the accuracy in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.eq(self.y_pred, y))
        else:
            raise NotImplementedError()

def sgd_optimization_gauss(learning_rate=0.13, n_epochs=1000,
                           batch_size=600):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    """
    #datasets = load_data(dataset)

    diary = Diary(name='experiment', path='results')
    diary.add_notebook('training')
    diary.add_notebook('validation')

    means=[[0,0],[5,5]]
    cov=[[[1,0],[0,1]],[[3,0],[0,3]]]
    samples=[4000,10000]
    datasets = generate_gaussian_data(means, cov, samples)

    diary.add_notebook('data')
    diary.add_entry('data', ['num_classes', len(samples)])
    diary.add_entry('data', ['means', means])
    diary.add_entry('data', ['covariance', cov])
    diary.add_entry('data', ['samples', samples])

    diary.add_entry('data', ['batch_size', batch_size])

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    diary.add_entry('data', ['train_size', len(train_set_y.eval())])
    diary.add_entry('data', ['valid_size', len(valid_set_y.eval())])
    diary.add_entry('data', ['test_size', len(test_set_y.eval())])

    pt = PresentationTier()
    pt.plot_samples(train_set_x.eval(), train_set_y.eval())

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    delta = 20
    x_min = numpy.min(train_set_x.eval(),axis=0)
    x_max = numpy.max(train_set_x.eval(),axis=0)
    x1_lin = numpy.linspace(x_min[0], x_max[0], delta)
    x2_lin = numpy.linspace(x_min[1], x_max[1], delta)

    MX1, MX2 = numpy.meshgrid(x1_lin, x2_lin)
    x_grid = numpy.asarray([MX1.flatten(),MX2.flatten()]).T
    grid_set_x = theano.shared(numpy.asarray(x_grid,
                                             dtype=theano.config.floatX),
                               borrow=True)
    n_grid_batches = grid_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    n_in = train_set_x.eval().shape[-1]
    n_out = max(train_set_y.eval()) + 1
    classifier = LogisticRegression(input=x, n_in=n_in, n_out=n_out)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # Scores
    grid_scores_model = theano.function(inputs=[],
            outputs=classifier.scores(),
            givens={
                x: grid_set_x})

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
    # compute the gradient of cost with respect to theta = (W,b)
    g_w = T.grad(cost=cost, wrt=classifier.w)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.w, classifier.w - learning_rate * g_w),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    # Accuracy
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

    # Loss
    training_error_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    validation_error_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    print('Creating error and accuracy vectors')
    error_train  = numpy.zeros(n_epochs+1)
    error_val = numpy.zeros(n_epochs+1)
    accuracy_train = numpy.zeros(n_epochs+1)
    accuracy_val = numpy.zeros(n_epochs+1)

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0
    CS = None
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                    (epoch, minibatch_index + 1, n_train_batches,
                    this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

        scores_grid = grid_scores_model()
        pt.update_contourline(grid_set_x.eval(), scores_grid, delta)
        scores_train = numpy.asarray([training_scores_model(i) for i
                                    in range(n_train_batches)]).flatten()
        scores_val = numpy.asarray([validation_scores_model(i) for i
                                  in range(n_valid_batches)]).flatten()
        fig = pt.plot_reliability_diagram(scores_train, train_set_y.eval(),
                                    scores_val, valid_set_y.eval())
        diary.save_figure(fig, filename='reliability_diagram', extension='svg')
        fig = pt.plot_histogram_scores(scores_train, scores_val)
        diary.save_figure(fig, filename='histogram_scores', extension='svg')

        # Performance
        accuracy_train[epoch] = numpy.asarray([training_accuracy_model(i) for i
                                in range(n_train_batches)]).flatten().mean()
        accuracy_val[epoch] = numpy.asarray([validation_accuracy_model(i) for i
                                in range(n_valid_batches)]).flatten().mean()
        error_train[epoch] = numpy.asarray([training_error_model(i) for i
                                in range(n_train_batches)]).flatten().mean()
        error_val[epoch] = numpy.asarray([validation_error_model(i) for i
                               in range(n_valid_batches)]).flatten().mean()

        diary.add_entry('training', [error_train[epoch], accuracy_train[epoch]])
        diary.add_entry('validation', [error_val[epoch], accuracy_val[epoch]])

        fig = pt.plot_accuracy(accuracy_train[1:epoch], accuracy_val[1:epoch])
        diary.save_figure(fig, filename='accuracy', extension='svg')
        fig = pt.plot_error(error_train[1:epoch], error_val[1:epoch], 'cross-entropy')
        diary.save_figure(fig, filename='error', extension='svg')


    pt.update_contourline(grid_set_x.eval(), scores_grid, delta,
            clabel=True)
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))

    #print >> sys.stderr, ('The code for file ' +
    #                      os.path.split(__file__)[1] +
    #                      ' ran for %.1fs' % ((end_time - start_time)))

if __name__ == '__main__':
    sgd_optimization_gauss(learning_rate=learning_rate, batch_size=batch_size,
            n_epochs=n_epochs)
