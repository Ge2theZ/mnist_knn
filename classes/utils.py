from classes.knn_lib import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class Utils():
    def plotKAccuracieDiagram(accuracies,klist):
        plt.plot(klist, accuracies)
        plt.suptitle('K-Accuracy Diagram')
        plt.ylabel('accuracy')
        plt.xlabel('K')
        plt.savefig('./images/kAccuracies.png')
        plt.show()

    def find_k(train, val, labelsTrain, labelsVal, trainSize, valSize):
        # initialize the values of k for our k-Nearest Neighbor classifier along with the
        # list of accuracies for each value of k
        kVals = range(1, 30, 2)
        accuracies = []
        # loop over various values of `k` for the k-Nearest Neighbor classifier
        for k in range(1, 30, 2):
            # train the k-Nearest Neighbor classifier with the current value of `k`
            # sklearn knn implementation
            model = KNeighborsClassifier(n_neighbors=k)
            # our knn implementation
            #model = knn(k=k)
            model.fit(train[:trainSize, :], labelsTrain[:trainSize])

            # evaluate the model and update the accuracies list
            score = model.score(val[:valSize, :], labelsVal[:valSize])
            print("k=%d, accuracy=%.2f%%" % (k, score * 100))
            accuracies.append(score)

        # find the value of k that has the largest accuracy
        i = int(np.argmax(accuracies))
        print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                               accuracies[i] * 100))
        Utils.plotKAccuracieDiagram(accuracies,kVals)
        return kVals[i], accuracies[i] * 100
