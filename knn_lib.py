from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math
import sys

def euclidian(a, b):
    return np.linalg.norm(a - b, ord=2)

def manhatten(a, b):
    return np.linalg.norm(a - b, ord=1)

class knn():
    def __init__(self, k=1, distanceFunc=euclidian):
        self.trainedData = []
        self.k = k
        self.distanceFunc = distanceFunc
        self.lengths = []

    def fit(self, data, label):
        for i in range(len(data)):
            self.trainedData.append((data[i],label[i]))

    def predict(self, samples):
        predictions = []
        for sample in samples:
            self.lengths = []
            nbs = self.getNeighbors(sample)
            labels = {}
            for i in range(self.k):
                if self.trainedData[self.lengths[i][1]][1] in labels:
                    labels[self.trainedData[self.lengths[i][1]][1]] += 1
                else:
                    labels[self.trainedData[self.lengths[i][1]][1]] = 1
            #sort dict by value
            labels = sorted(labels.items(), key=lambda x: x[1], reverse=True)
            #take first item in dict
            predictions.append(next(iter(labels))[0])
        return predictions

    def getNeighbors(self, sample):
        for idx, tupel in enumerate(self.trainedData):
             self.lengths.append((self.distanceFunc(tupel[0], sample),idx))
        self.lengths = sorted(self.lengths, key=lambda x: x[0])
        tmp = []
        for i in range(self.k):
            tmp.append(self.trainedData[self.lengths[i][1]])
        return tmp

    def score(self, valData, valLabel):
        right = 0
        predictions = self.predict(valData)
        for idx, prediction in enumerate(predictions):
            if(prediction == valLabel[idx]):
                right += 1
        return right/len(valData)


# --------------------- Testing/Beispiel ---------------------
'''
import sklearn
if int((sklearn.__version__).split(".")[1]) < 18:
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

# load the MNIST digits dataset
mnist = datasets.load_digits()
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25, random_state=42)

_knn = knn(k=8, distanceFunc=manhatten)
_knn.fit(trainData,trainLabels)
score = _knn.score(testData,testLabels)
print("\nAccuracy: {}".format(score))
'''
