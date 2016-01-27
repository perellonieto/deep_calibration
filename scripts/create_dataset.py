#!/usr/bin/python
import numpy as np
from os import path
from mnist import load_mnist

PATH_MNIST='../datasets/mnist/downloads/'
PATH_SAVE='../datasets/mnist/'
bin_threshold=0.5
salt_pepper_proportion=0.25

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

def add_gaussian_noise(data,mean,std):
    new_data = data + np.random.normal(mean,std,size=np.shape(data))
    new_data = new_data - np.min(new_data)
    new_data /= np.max(new_data)
    return new_data

# load data
if __name__ == "__main__": 
    for dataset in ['training', 'testing']:
        ret = load_mnist(path=PATH_MNIST, dataset=dataset)

        X = ret[0]
        Y = ret[1]

        bin_X = np.array(binaryze_dataset(X,bin_threshold),dtype=int)
        sap_bin_X = np.array(add_salt_and_pepper(bin_X,
            salt_pepper_proportion),dtype=bool)

        np.save(path.join(PATH_SAVE, "{}_X".format(dataset)), sap_bin_X)
        np.save(path.join(PATH_SAVE, "{}_Y".format(dataset)), Y)
