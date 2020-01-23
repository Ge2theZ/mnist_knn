from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

class knn():
    def __init__(self, k=1, distanceFunc="euclidean"):
        self.trainedData = []
        self.k = k
        self.distanceFunc = distanceFunc
        self.lengths = []

    def train(self, data, label):
        for i in range(len(data)):
            self.trainedData.append((data[i],label[i]))

    def predict(self, sample):
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
        #return first item in dict
        return next(iter(labels))

    def getNeighbors(self, sample):
        for idx, tupel in enumerate(self.trainedData):
             self.lengths.append((self.euclidian(tupel[0], sample),idx,tupel[1]))
        self.lengths = sorted(self.lengths, key=lambda x: x[0])
        tmp = []
        for i in range(self.k):
            tmp.append(self.trainedData[self.lengths[self.k][1]])
        return tmp

    def euclidian(self, a, b):
        return np.linalg.norm(a-b)

    def score(self):
        pass

# --------- Testing ---------
import sklearn
if int((sklearn.__version__).split(".")[1]) < 18:
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

# load the MNIST digits dataset
mnist = datasets.load_digits()
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                                  mnist.target, test_size=0.25, random_state=42)
_knn = knn(k=22)
_knn.train(trainData,trainLabels)

for i in range(10):
    randomSample = random.randint(0, len(testData))
    prediction = _knn.predict(testData[randomSample])
    print("The number was {} and the prediction was {}".format(testLabels[randomSample],prediction[0]))
