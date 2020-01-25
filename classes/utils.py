from classes.knn_lib import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import datetime

class Utils():
    def plotKAccuracieDiagram(accuracies,klist, name):
        plt.plot(klist, accuracies)
        plt.suptitle(name)
        plt.ylabel('accuracy')
        plt.xlabel('K')
        plt.savefig('./images/kAccuracies_k1_37800samples_sklearn.png')
        plt.show()

    def find_k(train, val, labelsTrain, labelsVal, trainSize, valSize, description):
        # initialize the values of k for our k-Nearest Neighbor classifier along with the
        # list of accuracies for each value of k
        kVals = range(1, 30, 2)
        accuracies = []
        # loop over various values of `k` for the k-Nearest Neighbor classifier
        for k in range(1, 30, 2):
            # train the k-Nearest Neighbor classifier with the current value of `k`
            # sklearn knn implementation
            #print("{} Initializing model with k={}. ".format(datetime.datetime.now(), k))
            model = KNeighborsClassifier(n_neighbors=k)
            #print("{} Initialized model with k={}. ".format(datetime.datetime.now(), k))
            # our knn implementation
            #model = knn(k=k)
            #print("{} Fitting model with k={}. ".format(datetime.datetime.now(), k))
            model.fit(train[:trainSize, :], labelsTrain[:trainSize])
            #print("{} Fitted model with k={}. ".format(datetime.datetime.now(), k))

            # evaluate the model and update the accuracies list
            #print("{} Evaluating model with k={} and a validation data set of size = {}. ".format(datetime.datetime.now(), k, valSize))
            score = model.score(val[:valSize, :], labelsVal[:valSize])
            #print("{} Evaluated model with k={} and a validation data set of size = {}. ".format(datetime.datetime.now(), k, valSize))
            print("{} {}: k={}, accuracy={}%".format(datetime.datetime.now(), description,k, score * 100))
            accuracies.append(score)

        # find the value of k that has the largest accuracy
        i = int(np.argmax(accuracies))
        print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                               accuracies[i] * 100))
        Utils.plotKAccuracieDiagram(accuracies,kVals, description)
        return kVals[i], accuracies[i] * 100
