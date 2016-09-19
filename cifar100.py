import numpy as np
import _pickle as pickle
import os
import scipy
from knn import NearestNeighbour
num_of_train = 50000
num_of_test = 10000


def load_CIFAR_test(file):
    """ load single batch of cifar"""
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        print (datadict.keys())
        X = datadict['data']
        Y = datadict['fine_labels']
        Z = datadict['coarse_labels']
        X = X.reshape(num_of_test, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        Z = np.array(Z)
    return X, Y, Z


def load_CIFAR_train(file):
    """ load single batch of cifar"""
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        print (datadict.keys())
        X = datadict['data']
        Y = datadict['fine_labels']
        Z = datadict['coarse_labels']
        X = X.reshape(num_of_train, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        Z = np.array(Z)
    return X, Y, Z


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    zs = []
    f = os.path.join(ROOT, 'train')
    X, Y, Z = load_CIFAR_train(f)
    xs.append(X)
    ys.append(Y)
    zs.append(Z)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Ztr = np.concatenate(zs)
    del X, Y, Z
    """
    Xtr, Ytr, Ztr = training data
    Xte, Yte, Zte = testing data
    """
    Xte, Yte, Zte = load_CIFAR_test(os.path.join(ROOT, 'test'))
    return Xtr, Ytr, Ztr, Xte, Yte, Zte

def run_knn():
    Xtr, Ytr, Ztr, Xte, Yte, Zte = load_CIFAR10('cifar-100-python')

    # flattens out all images to be one dimensional
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows become 50000x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xtr_rows become 10000x 3072 

if __name__ == '__main__':
    run_knn()
