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

class knn():
    def _init_(selfs, k=1):
        self.data = []
        self.k = k

    def train(self, data, label):
        for i in range(len(data)):
            self.data.append((data[i],label[i]))

    def predict(self, data):
        pass

    def euclidian(self, a, b):
        return numpy.linalg.norm(a-b)



training_samples = int(len(mnist_data)*(training_percentage))
training_data = np.zeros((training_samples,mnist.data.shape[1]))
training_data[:] = mnist_data[:training_samples]
