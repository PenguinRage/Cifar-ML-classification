N = len(y)

pos_of_class0 = np.where(y==0)[0] #positions of class 0
pos_of_class1 = np.where(y==1)[0] #positions of class 1
pos_of_class2 = np.where(y==2)[0] #positions of class 2

pl.scatter(X[pos_of_class0,0], X[pos_of_class0,1], c='r', edgecolor='') #class = 0
pl.scatter(X[pos_of_class1,0], X[pos_of_class1,1], c='g', edgecolor='') #class = 1
pl.scatter(X[pos_of_class2,0], X[pos_of_class2,1], c='b', edgecolor='') #class = 2
