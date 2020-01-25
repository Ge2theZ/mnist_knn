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

# Any results you write to the current directory are saved as output.
train = pd.read_csv('data/train.csv')
submission = pd.read_csv('data/test.csv')

y_train = train['label']
X_train = train.drop('label', axis=1)
X_submission = submission

y_train.head()
X_submission.head()
y_train = y_train.to_numpy()
X_train = X_train.to_numpy()

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


# plot data from tsme
tsne_result = TSNE(n_components=2).fit_transform(X_train[:1000, :])
plt.scatter(tsne_result[:4000, 0], tsne_result[:1000, 1], c=y_train[:1000], edgecolor='none', alpha=0.5,
            cmap=plt.get_cmap('jet', 10), s=5)
plt.colorbar()
plt.title("t-sne")
plt.show()


pca = PCA(n_components=50)
X_train_transformed = pca.fit_transform(X_train)
X_submission_transformed = pca.transform(X_submission)

transformer = random_projection.GaussianRandomProjection(n_components=50)
X_train_transformer = transformer.fit_transform(X_train)
X_test_transformer = transformer.fit_transform(X_submission)

tsne = TSNE(n_components=2)
X_train_tsne = tsne.fit_transform(X_train)
X_submission_tsne = tsne.fit_transform(X_submission)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_train_transformed, y_train, test_size=0.05, random_state=13)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X_train, y_train, test_size=0.05, random_state=13)
X_train_rp, X_test_rp, y_train_rp, y_test_rp = train_test_split(X_train_transformer, y_train, test_size=0.05, random_state=13)
X_train_tsne, X_test_tsne, y_train_tnse, y_test_tnse = train_test_split(X_train_tsne, y_train, test_size=0.05, random_state=13)


print("X_train_pca shape: ", X_train_pca.shape)
print("X_test_pca shape: ", X_test_pca.shape)

print("X_train_raw shape: ", X_train_raw.shape)
print("X_test_raw shape: ", X_test_raw.shape)

trainSize = 35910
testSize = 1890

(k_rp, percent) = Utils.find_k(X_train_rp, X_test_rp, y_train_rp, y_test_rp, trainSize, testSize, "Random Projection")
(k_pca, percent) = Utils.find_k(X_train_pca, X_test_pca, y_train_pca, y_test_pca, trainSize, testSize, "Principal Component")
(k_tsne, percent) = Utils.find_k(X_train_tsne, X_test_tsne, y_train_tnse, y_train_tnse, trainSize, testSize, "T-SNE")
#(k_raw, percent) = Utils.find_k(X_train_raw, X_test_raw, y_train_raw, y_test_raw, trainSize, valSize, "Raw MNIST Data")


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