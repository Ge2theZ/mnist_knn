from sklearn import datasets
from . import knn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

training_percentage = 0.9 #90%

# load the MNIST digits dataset
mnist = datasets.load_digits()
# Uses array from numpy for the mnist dat to convert to 32bit integer
mnist_data = np.array(mnist.data.shape, dtype=np.int32)
mnist_data = np.copy(mnist.data.astype(np.int32))

training_samples = int(len(mnist_data)*(training_percentage))
training_data = np.zeros((training_samples,mnist.data.shape[1]))
training_data[:] = mnist_data[:training_samples]


class knn():
    def _init_(selfs, k=1):
        self.data = []
        self.k = k
        self.lengths = []

    def train(self, data, label):
        for i in range(len(data)):
            self.data.append((data[i],label[i]))

    def predict(self, sample):
        nbs = self.getNeighbors(sample)
        labels = {}
        for i in range(k):
            labels[self.data[self.lengths[k][0]][1]] += 1

    def getNeighbors(self, sample):
        for idx, tupel in enumerate(self.data):
             self.lengths.append((self.euclidian(tupel[0], sample),idx))
        self.lengths.sort(key=lambda x: x[0])
        tmp = []
        for i in range(self.k):
            tmp.append(self.data[self.lengths[k][1]])
        return tmp



    def euclidian(self, a, b):
        return numpy.linalg.norm(a-b)
