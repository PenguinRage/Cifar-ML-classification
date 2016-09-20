import numpy as np
import _pickle as pickle
import os
import scipy
from cifar100_knn import NearestNeighbour
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

def load_CIFAR10_image(image):
    X = np.array(image)
    X = X.reshape(1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    return X

def load_CIFAR10_images(TROOT,ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    zs = []
    f = os.path.join(TROOT, 'train')
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
    Xte = testing data
    """
    xs = []
    for filename in glob.glob(ROOT + '/*.png'):
        image = Image.open(filename)
        X = load_CIFAR10_image(image)
        xs.append(X)
    Xte = np.concatenate(xs)
    return Xtr, Ytr, Ztr, Xte

def plot_confusion_matrix(cm, title,i, cmap=plt.cm.Blues):
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('CIFAR_10_confusion_matrix_'+ i + '.png')


def results(Y_pred, Yte):
    cm = confusion_matrix(Yte, Y_pred)
    title = "10NN Confusion Matrix"
    i= "normal"
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm,title, i)


def run_knn():
        new = input('Testing with new undefined images? (y or n): ')
    if (new == 'y'):
        Xtr, Ytr, Xte = load_CIFAR10_images('cifar-100-python','INFO3406_assignment1_query')

    else:
        Xtr, Ytr, Ztr, Xte, Yte, Zte = load_CIFAR10('cifar-100-python')

    # flattens out all images to be one dimensional
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows become 50000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xtr_rows become 10000 x 3072

    for k in [10]:
        # create NearestNeighbour 
        nn = NearestNeighbour()
        # Train data
        nn.train(Xtr_rows)
        # Predict the values of fine and coarse labels
        Yte_predict, Zte_predict = nn.predict(Xte_rows, k)
        
        if (new != 'y'):
            fine_acc = np.mean(Yte_predict == Yte)
            coarse_acc = np.mean(Zte_predict == Yte)
            print('K-NN %d' % (k))
            print('fine label accuracy: %f' % (fine_acc))
            print('coarse label accuracy: %f' % (coarse_acc))
    
    # Save output to csv
    np.savetxt("cifar100_predictions.csv",(Zte_predict, Yte_predict), delimiter=",")

if __name__ == '__main__':
    run_knn()
