"""
Source: https://gurus.pyimagesearch.com/lesson-sample-k-nearest-neighbor-classification/#
Comment: Made some minor adjustments
"""

# import the necessary packages
from __future__ import print_function
from sklearn.metrics import classification_report
from sklearn import random_projection
from skimage import exposure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import numpy as np
import imutils
import cv2
import sklearn
from classes.knn_lib import *
from classes.utils import *
import pandas as pd

mnistDataAmount = 5000
testPercentage = 0.2

# Any results you write to the current directory are saved as output.
train = pd.read_csv('data/train.csv')

mnist_labels = train['label']
mnist_data = train.drop('label', axis=1)
mnist_labels.head()

mnist_labels = mnist_labels.to_numpy()[:mnistDataAmount]
mnist_data = mnist_data.to_numpy()[:mnistDataAmount]

train_data_raw, test_data_raw, train_label_raw, test_label_raw = train_test_split(mnist_data, mnist_labels, test_size=testPercentage)

'''
# Plot data for pca
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_train)
plt.scatter(pca_result[:4000, 0], pca_result[:4000, 1], c=y_train[:4000], edgecolor='none', alpha=0.5,
            cmap=plt.get_cmap('jet', 10), s=5)
plt.colorbar()
plt.title("PCA")
plt.show()

# plot data for random projection
transformer = random_projection.GaussianRandomProjection(n_components=2)
rand_projection_result = transformer.fit_transform(X_train)
plt.scatter(rand_projection_result[:4000, 0], rand_projection_result[:4000, 1], c=y_train[:4000], edgecolor='none', alpha=0.5,
            cmap=plt.get_cmap('jet', 10), s=5)
plt.colorbar()
plt.title("Random Transformer")
plt.show()

# plot data from tsne
tsne_result = TSNE(n_components=2).fit_transform(X_train[:1000, :])
plt.scatter(tsne_result[:4000, 0], tsne_result[:1000, 1], c=y_train[:1000], edgecolor='none', alpha=0.5,
            cmap=plt.get_cmap('jet', 10), s=5)
plt.colorbar()
plt.title("t-sne")
plt.show()
'''
print("Creating PCA Representation")
pca = PCA(n_components=50)
data_pca = pca.fit_transform(mnist_data)
train_data_pca, test_data_pca, train_label_pca, test_label_pca = train_test_split(data_pca, mnist_labels, test_size=testPercentage, random_state=13)

print("Creating RP Representation")
transformer = random_projection.GaussianRandomProjection(n_components=50)
data_rp = transformer.fit_transform(mnist_data)
train_data_rp, test_data_rp, train_label_rp, test_label_rp = train_test_split(data_rp, mnist_labels, test_size=testPercentage, random_state=13)


'''print("Creating TSNE Representation")
tsne = TSNE(n_components=2)
data_tsne = tsne.fit_transform(mnist_data)
train_data_tsne, test_data_tsne, train_label_tsne, test_label_tsne = train_test_split(data_tsne, mnist_labels, test_size=testPercentage)
'''

(k_pca, percent) = Utils.find_k(train_data_pca, train_label_raw, test_data_pca, test_label_raw, "Principal Component")
(k_rp, percent) = Utils.find_k(train_data_rp, train_label_raw, test_data_rp, test_label_raw, "Random Projection")
#(k_tsne, percent) = Utils.find_k(train_data_tsne, train_label_raw, test_data_tsne, test_label_raw, "T-SNE")
(k_raw, percent) = Utils.find_k(train_data_raw, train_label_raw, test_data_raw, test_label_raw, "Raw MNIST Data")

'''
# fit knn with pca
model = knn(k=k_pca)
model.fit(X_train_pca, y_train_pca)
print("EVALUATION ON TESTING DATA FOR PCA")
predictions = model.predict(X_test_pca[:100,:])
print(classification_report(X_test_pca[:100,:], predictions))
'''
'''
# re-train our classifier using the best k value and predict the labels of the
# test data
print("{} Reinitialize model with k={}. ".format(datetime.datetime.now(), k_raw))
model = KNeighborsClassifier(n_neighbors=k_raw)
print("{} Reinitialized model with k={}. ".format(datetime.datetime.now(), k_raw))
#model = knn(k=k_raw)
print("{} Fitting final model with k={}. ".format(datetime.datetime.now(), k_raw))
model.fit(trainArr[:trainSize, :], labelsTrainArr[:trainSize])
#predictions = model.predict(valArr[:valSize,:])
print("{} Fitted final model with k={}. ".format(datetime.datetime.now(), k_raw))

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
#print("EVALUATION ON TESTING DATA")
#print(classification_report(labelsValArr[:valSize], predictions))

# loop over a few random digits
for i in list(map(int, np.random.randint(0, high=valSize, size=(5,)))):
    # grab the image and classify it
    image = testArr[i]
    prediction = model.predict(image.reshape(1, -1))[0]

    # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
    # then resize it to 32 x 32 pixels so we can see it better
    image = image.reshape((28, 28)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=128, inter=cv2.INTER_CUBIC)

    # show the prediction
    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
'''
