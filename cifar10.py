from PIL import Image
import numpy as np
import _pickle as pickle
import os
import glob
# only using the library for confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from knn import NearestNeighbour

# Loads a binary patch file
def load_CIFAR_batch(file):
    """ load single batch of cifar"""
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
    return X, Y

# Loads training and test batch files 
def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for i in range(1,6):
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

# converts image into array and formats it into the same format as training
# Note: Due to evaluation and being vague this might need to edited to your specifications
def load_CIFAR10_image(image):
    X = np.array(image)
    X = X.reshape(1, 3, 32, 32).transpose(0,2,3,1).astype("float")
    return X

# Loads image and formats it 
def load_CIFAR10_images(TROOT,ROOT):
    xs = []
    ys = []
    for i in range(1,6):
        f = os.path.join(TROOT, 'data_batch_%d' % (i,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    """
    Xtr, Ytr = training data
    Xte = testing data
    """
    xs = []
    for filename in glob.glob(ROOT + '/*.png'):
        image = Image.open(filename)
        X = load_CIFAR10_image(image)
        xs.append(X)
    Xte = np.concatenate(xs)
    return Xtr, Ytr, Xte

# prints out confusion matrix
def plot_confusion_matrix(cm, title,i, cmap=plt.cm.Blues):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('CIFAR_10_confusion_matrix_'+ i + '.png')

# prints our results into a nice confusion matrix
def results(Y_pred, Yte):
    cm = confusion_matrix(Yte, Y_pred)
    title = "10NN Confusion Matrix"
    i= "normal"
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm,title, i)
    # title = "10NN Normalised Confusion Matrix"
    # i = "normalised"
    # cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # plt.figure()
    # plot_confusion_matrix(cm_norm, title, i)


# The main knn function that uses the NearestNeighbour class in knn.py
def run_knn():
    new = input('Testing with new undefined images? (y or n): ')
    if (new == 'y'):
        Xtr, Ytr, Xte = load_CIFAR10_images('cifar-10-batches-py','INFO3406_assignment1_query')

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
        if (new != 'y'):
            # Compute the accuracy between the actuals and the predictions
            acc = np.mean(Yte_predict == Yte)
            print('K-NN %d' % (k))
            print('accuracy: %f' % (acc))
            # For graphing
            # results(Yte_predict,Yte)
        
        # Print predictions to csv
    np.savetxt("cifar10_predictions.csv", Yte_predict, delimiter=",")

    # return the predictions and the actual values
    return


if __name__ == '__main__': 
    run_knn()
