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

"""
def results(Y_pred,Z_pred, Yte ,Zte):
    # seperate and print out class results

    #correct and incorrect counts
    correct = 0
    incorrect = 0


    # there are 20 super classes
    super = np.zeros(20)
    inSuper = np.zeros(20)
    superLabels = ['aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables', 'household electrical devices'
        , 'household furniture', 'insects', 'large carnivores','large man - made outdoor things', 'large natural outdoor scenes'
        , 'large omnivores and herbivores', 'medium-sized mammals', 'non-insect invertabrates', 'people', 'reptiles', 'small mammals'
        , 'trees', 'vehicles 1', 'vehicles 2']

    # and 100 sub (or normal) classes
    sub = np.zeroes(100)
    inSub = np.zeroes(100)
    subLabels = ['beaver', 'dolphin', 'otter', 'seal', 'whale'
        ,'aquarium fish', 'flatfish', 'ray', 'shark, trout'
        ,'orchids', 'poppies', 'roses', 'sunflowers',' tulips'
        ,'bottles', 'bowls', 'cans', 'cups', 'plates'
        ,'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers'
        ,'clock', 'computer keyboard', 'lamp', 'telephone', 'television'
        ,'bed', 'chair', 'couch', 'table', 'wardrobe'
        ,'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'
        ,'bear', 'leopard', 'lion', 'tiger', 'wolf'
        ,'bridge', 'castle', 'house', 'road', 'skyscraper'
        ,'cloud', 'forest', 'mountain', 'plain', 'sea'
        ,'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'
        ,'fox', 'porcupine', 'possum', 'raccoon', 'skunk'
        ,'crab', 'lobster', 'snail', 'spider', 'worm'
        ,'baby', 'boy', 'girl', 'man', 'woman'
        ,'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'
        ,'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'
        ,'maple', 'oak', 'palm', 'pine', 'willow'
        ,'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train'
        ,'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor']



    # a correct prediction is only when Yte and Zte are predicted correctly
    testSize = 10000

    for i in range (testSize):
        #correct prediction
        if Y_pred[i]==Yte[i]:
            super[Yte[i]] += 1
            if Z_pred[i]==Zte[i]:
                sub[Zte[i]] += 1
                correct +=1
            else:
                #incorrect 2nd lable
                inSub[Zte[i]] +=1
                incorrect +=1

        elif Y_pred[i]!=Yte[i]:
            # incorrect prediction
            inSuper[Y_pred[i]]
            incorrect+=1
            if Z_pred[i]==Zte[i]:
                sub[Zte[i]] += 1
            else:
                # incorrect 2nd lable
                inSub[Zte[i]] += 1

    #after for loop

    # super : contains counts of all correct super class predictions
    # sub : contains counts of all correct sub class predictions

    # inSuper : contains counts of all incorrect super class predictions
    # inSub : contains counts of all incorrect sub class predictions
"""




def run_knn():
    Xtr, Ytr, Ztr, Xte, Yte, Zte = load_CIFAR10('cifar-100-python')

    # flattens out all images to be one dimensional
    Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3)  # Xtr_rows become 50000 x 3072
    Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3)  # Xtr_rows become 10000 x 3072

    validation_accuracies = []
    for k in [10]:
        nn = NearestNeighbour()
        nn.train(Xtr_rows, Ytr, Ztr)
        Yte_predict, Zte_predict = nn.predict(Xte_rows, k, Yte, Zte)
        fine_acc = np.mean(Yte_predict == Yte)
        coarse_acc = np.mean(Zte_predict == Yte)
        print('K-NN %d' % (k))
        print('fine label accuracy: %f' % (fine_acc))
        print('coarse label accuracy: %f' % (coarse_acc))

if __name__ == '__main__':
    run_knn()
