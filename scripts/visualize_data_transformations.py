#!/usr/bin/python
import numpy as np
from mnist import load_mnist
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (3,3)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
plt.ion()

def binaryze_dataset(data, threshold=0.5):
    new_data = np.zeros(np.shape(data))
    new_data[data > threshold] = 1
    return new_data

# Given a dataset with samples and features, it chooses some features for each
# sample from following a Bernoulli distribution and inverts their values
def add_salt_and_pepper(data, proportion):
    num_samples = np.shape(data)[0]
    original_shape = np.shape(data[0])

    new_data = np.reshape(data, (num_samples,-1))
    num_features = np.shape(new_data)[1]

    salt_and_pepper = np.random.binomial(1, proportion,
                                         size=(num_samples,num_features))

    new_data = (1-new_data)*salt_and_pepper + new_data*(1-salt_and_pepper)
    return np.reshape(new_data, (num_samples, original_shape[0],
        original_shape[1]))

# load data
ret = load_mnist(path='../datasets/mnist/downloads/', digits=[0,1,2])

X = ret[0]
Y = ret[1]

# show one example
fig = plt.figure('mnist_example')
for i in range(3):
    plt.subplot(1,3,i)
    sample = X[i]
    plt.imshow(sample)
    plt.show()

sap_X = add_salt_and_pepper(X, 0.25)
fig = plt.figure('mnist_salt_and_pepper')
for i in range(3):
    plt.subplot(1,3,i)
    sample = sap_X[i]
    plt.imshow(sample)
    plt.show()

bin_X = binaryze_dataset(X,0.5)
fig = plt.figure('mnist_binarized')
for i in range(3):
    plt.subplot(1,3,i)
    sample = bin_X[i]
    plt.imshow(sample)
    plt.show()

sap_bin_X = add_salt_and_pepper(bin_X, 0.25)
fig = plt.figure('binarized_salt_and_pepper')
for i in range(3):
    plt.subplot(1,3,i)
    sample = sap_bin_X[i]
    plt.imshow(sample)
    plt.show()
