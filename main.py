import numpy as np
import _pickle as pickle
import os
import scipy
from knn import NearestNeighbour


def load_CIFAR_batch(file):
    """ load single batch of cifar"""
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
    return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for i in range(3, 4):
        f = os.path.join(ROOT, 'data_batch_%d' % (i,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    """
    Xtr, Ytr = training data
    Xte, Yte = testing data
    """
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def results(Y_pred, Yte):
    inPred = np.zeros((10, 11))
    # set 1 to 10 values(0 to 9)
    for i in range(10):
        inPred[i, 0] = i

    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # in testing each class has 1000

    #Yte needs to be set to the test size
    # k = len(Yte)
    k = 10

    correct = np.zeros(10)
    incorrect = np.zeros(10)

    # count correct and incorrect predictions
    for i in range(k):
        if Y_pred[i] == Yte[i]:
            correct[Yte[i]] = correct[Yte[i]] + 1
        else:
            incorrect[Yte[i]] = incorrect[Yte[i]] + 1
            inPred[Yte[i], Y_pred[i] + 1] += 1

    print("Each class has a total of 1000")
    print("Percentage of Correct and Incorrect Classifications by class")
    # print out the percentage of correct predictions
    for i in range(len(correct)):
        wrong = (incorrect[i] / 1000) * 100
        right = (correct[i] / 1000) * 100
        print(labels[i] + " correct: " + str(right) + " incorrect: " + str(wrong))

    print("Number of times each class was predicted wrongfully per class:")
    print ()

    for i in range (10):
        print(labels[i].upper() +":")
        print(labels[0] +"  "+ labels[1] +"  "+ labels[2] +"  "+ labels[3] +"  "+ labels[4] +"  "+ labels[5] +"  "+ labels[6] +"  "+ labels[7] +"  "+ labels[8] +"  "+ labels[8] )
        print(str(inPred[i,1]) +"    "+ str(inPred[i,2]) +"  "+ str(inPred[i,3]) +"   "+ str(inPred[i,4]) +"   "+ str(inPred[i,5]) +"   "+ str(inPred[i,6]) +"   "+ str(inPred[i,7]) +"   "+ str(inPred[i,8]) +"   "+ str(inPred[i,9]) +"   "+ str(inPred[i,10]))
        print()



def run_knn():
    Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py')

    # flattens out all images to be one dimensional
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows become 50000x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xtr_rows become 10000x 3072

    validation_accuracies = []
    for k in [10]:
        nn = NearestNeighbour()  # create a Nearest Neighbor classifier class
        nn.train(Xtr_rows, Ytr)  # train the classifier on the training images and labels
        Yte_predict = nn.predict(Xte_rows, k, Yte)  # predict labels on the test images
        # and now print the classification accuracy, which is the average number
        # of examples that are correctly predicted (i.e label matches)
        acc = np.mean(Yte_predict == Yte)
        print('K-NN %d' % (k))
        print('accuracy: %f' % (acc))

        validation_accuracies.append((k, acc))

    # return the predictions and the actual values
    return Yte_predict, Yte


if __name__ == '__main__':
    Y_pred, Yte = run_knn()
    results(Y_pred, Yte)
