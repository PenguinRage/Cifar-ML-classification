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

    def predict(self, X, K):
        
        # X is N * D where each row is an example we wish to predict label for
        num_test = X.shape[0]

        
        # Lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # Loop over all test rows
        for i in range(num_test):
            # using the L1 distance (sum of absolute)
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            
            # sort the distance
            min_index = np.argsort(distances)

            # K-Nearest component
            classes = np.zeros(10)

            for j in range(K):
                if self.ytr[min_index[j]] == 0:
                    classes[0] += 1
                elif self.ytr[min_index[j]]==1:
                    classes[1] += 1
                elif self.ytr[min_index[j]]==2:
                    classes[2] += 1
                elif self.ytr[min_index[j]]==3:
                    classes[3] += 1
                elif self.ytr[min_index[j]]==4:
                    classes[4] += 1
                elif self.ytr[min_index[j]]==5:
                    classes[5] += 1
                elif self.ytr[min_index[j]]==6:
                    classes[6] += 1
                elif self.ytr[min_index[j]]==7:
                    classes[7] += 1
                elif self.ytr[min_index[j]]==8:
                    classes[8] += 1
                elif self.ytr[min_index[j]]==9:
                    classes[9] += 1
                else:
                    print('Error - Invalid class')
            
            # predict the label of the nearest example
            Ypred[i] = np.argmax(classes)
            
            # print(Ypred[i])
        return Ypred
