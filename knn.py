from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

training_examples = 100

mnist = fetch_openml('mnist_784', data_home="./")
# Uses array from numpy for the mnist dat to convert to 32bit integer
mnist_data = np.array(mnist.data.shape, dtype=np.int32)
mnist_data = np.copy(mnist.data.astype(np.int32))

def knn(k):
    print(mnist.data.shape[1])
    training_data = np.zeros(training_examples,mnist.data.shape[1])
    training_data[:] = mnist_data[:training_examples]

knn(2)
