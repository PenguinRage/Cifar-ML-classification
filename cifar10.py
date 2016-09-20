import numpy as np
import _pickle as pickle
import os
from sklearn.metric import confusion_matrix
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
    for i in range(1, 6):
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

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('CIFAR_10_confusion_matrix')+'.jpg')


def results(Y_pred, Yte):
    labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cm = confusion_matrix(Yte, Y_pred,labels)
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)


def run_knn():
    new = input('Testing with a new batch? y or n')
    if (new == y):
        print("Hello")

    else:
        Xtr, Ytr, Xte, Yte = load_CIFAR10('cifar-10-batches-py')
        # flattens out all images to be one dimensional
        Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows become 50000x 3072
        Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xtr_rows become 10000x 3072
        validation_accuracies = []
        for k in [10]:
            nn = NearestNeighbour()  # create a Nearest Neighbor classifier class
            nn.train(Xtr_rows, Ytr)  # train the classifier on the training images and labels
            Yte_predict = nn.predict(Xte_rows, k)  # predict labels on the test images
            # and now print the classification accuracy, which is the average number
            # of examples that are correctly predicted (i.e label matches)
            acc = np.mean(Yte_predict == Yte)
            print('K-NN %d' % (k))
            print('accuracy: %f' % (acc))
        results(Yte_predict,Yte)

    # Print predictions to csv
    np.savetxt("cifar10_predictions.csv", Yte_predict, delimiter=",")
    # return the predictions and the actual values
    return Yte_predict, Yte


if __name__ == '__main__': 
    run_knn()
