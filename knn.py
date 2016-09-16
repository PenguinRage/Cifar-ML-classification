import numpy as np

class NearestNeighbour(object):
    def __init__(self):
        pass

    def train(self, X, y):
        
        # X is N*D where each row is an example.
        # Y is 1-dimension of size N
        # the nearest neighbour classifier simply remembers 
        # all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        
        # X is N * D where each row is an example we wish to predict label for
        num_test = X.shape[0]

        
        # Lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # Loop over all test rows
        for i in range(num_test):
            # using the L1 distance (sum of absolute)
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            
            # get the index with smallest distance
            min_index = np.argmin(distances)
            # predict the label of the nearest example
            Ypred[i] = self.ytr[min_index] 
        return Ypred
