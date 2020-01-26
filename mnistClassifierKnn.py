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

#train.csv has 37800 samples
mnistDataAmount =  1000 #30240
testPercentage = 0.2
testSize = 50 # defines the amount of digits to be used to find the best value of k within the test dataset
isSubmission = False # when true, then the final test.csv data is being klassified for the submission

# Any results you write to the current directory are saved as output.
train = pd.read_csv('data/train.csv')

mnist_labels = train['label']
mnist_data = train.drop('label', axis=1)
mnist_labels.head()

mnist_labels = mnist_labels.to_numpy()[:mnistDataAmount]
mnist_data = mnist_data.to_numpy()[:mnistDataAmount]

train_data_raw, test_data_raw, train_label_raw, test_label_raw = train_test_split(mnist_data, mnist_labels, test_size=testPercentage)


# Plot data for pca
pca = PCA(n_components=2)
pca_result = pca.fit_transform(mnist_data)
plt.scatter(pca_result[:4000, 0], pca_result[:4000, 1], c=mnist_labels[:4000], edgecolor='none', alpha=0.5,
            cmap=plt.get_cmap('jet', 10), s=5)
plt.colorbar()
plt.title("PCA")
plt.show()

# plot data for random projection
transformer = random_projection.GaussianRandomProjection(n_components=2)
rand_projection_result = transformer.fit_transform(mnist_data)
plt.scatter(rand_projection_result[:4000, 0], rand_projection_result[:4000, 1], c=mnist_labels[:4000], edgecolor='none', alpha=0.5,
            cmap=plt.get_cmap('jet', 10), s=5)
plt.colorbar()
plt.title("Random Transformer")
plt.show()

# plot data from tsne
tsne_result = TSNE(n_components=2).fit_transform(mnist_data[:1000, :])
plt.scatter(tsne_result[:4000, 0], tsne_result[:1000, 1], c=mnist_labels[:1000], edgecolor='none', alpha=0.5,
            cmap=plt.get_cmap('jet', 10), s=5)
plt.colorbar()
plt.title("t-sne")
plt.show()

print("Creating PCA Representation")
pca = PCA(n_components=50)
data_pca = pca.fit_transform(mnist_data)
train_data_pca, test_data_pca, train_label_pca, test_label_pca = train_test_split(data_pca, mnist_labels, test_size=testPercentage, random_state=13)

print("Creating RP Representation")
transformer = random_projection.GaussianRandomProjection(n_components=50)
data_rp = transformer.fit_transform(mnist_data)
train_data_rp, test_data_rp, train_label_rp, test_label_rp = train_test_split(data_rp, mnist_labels, test_size=testPercentage, random_state=13)

'''
print("Creating TSNE Representation")
tsne = TSNE(n_components=2)
data_tsne = tsne.fit_transform(mnist_data)
train_data_tsne, test_data_tsne, train_label_tsne, test_label_tsne = train_test_split(data_tsne, mnist_labels, test_size=testPercentage)
'''
(k_pca, comp_pca, percent) = Utils.find_k_with_components(train_data_pca, train_label_pca, test_data_pca, test_label_pca, "Principal Component")
(k_rp, comp_rp, percent) = Utils.find_k_with_components(train_data_rp, train_label_rp, test_data_rp, test_label_rp, "Random Projection")
#(k_tsne, percent) = Utils.find_k(train_data_tsne, train_label_tsne, test_data_tsne, test_label_tsne, "T-SNE")
(k_raw, percent) = Utils.find_k(train_data_raw, train_label_raw, test_data_raw, test_label_raw, testSize, "Raw MNIST Data")


#klassifier scores
scores = [] # 0 = raw / 1 = pca / 2 = rp

# re-train our classifier for RAW features with best value for K
model_raw = knn(k=k_raw)
model_raw.fit(train_data_raw, train_label_raw)
predictions_raw = model_raw.predict(test_data_raw)
print("###### Report for raw data ######")
print(classification_report(test_label_raw, predictions_raw))
Utils.plotConfusionMatrix(predictions_raw,test_label_raw, "RawDatawithElements{}".format(mnistDataAmount))
scores.append(Utils.getScore(predictions_raw, test_label_raw))

# re-train our classifier for pca features with best value for K and components
model_pca = knn(k=k_pca)
model_pca.fit(train_data_pca[:,:comp_pca], train_label_pca)
predictions_pca = model_pca.predict(test_data_pca[:,:comp_pca])
print("###### Report for pca data ######")
print(classification_report(test_label_pca, predictions_pca))
Utils.plotConfusionMatrix(predictions_pca,test_label_pca, "PCADatawithElements{}".format(mnistDataAmount))
scores.append(Utils.getScore(predictions_pca, test_label_pca))

# re-train our classifier for rp features with best value for K and components
model_rp = knn(k=k_rp)
model_rp.fit(train_data_rp[:,:comp_rp], train_label_rp)
predictions_rp = model_rp.predict(test_data_rp[:,:comp_rp])
print("###### Report for pca data ######")
print(classification_report(test_label_rp, predictions_rp))
Utils.plotConfusionMatrix(predictions_rp,test_label_rp, "RpDatawithElements{}".format(mnistDataAmount))
scores.append(Utils.getScore(predictions_rp, test_label_rp))


# make final classification on submission dataset
if(isSubmission):

    submission = pd.read_csv('data/test.csv')
    indexHighestKlassifier = scores.index(max(scores))
    features = ""

    if(indexHighestKlassifier == 0):
        features = "raw"
        model_raw = knn(k=k_raw)
        model_raw.fit(train_data_raw, train_label_raw)
        predict_labels = model_raw.predict(submission)
    if(indexHighestKlassifier == 1):
        features = "pca"
        pca = PCA(n_components=comp_pca)
        submission_pca = pca.fit_transform(submission)

        model_pca = knn(k=k_pca)
        model_pca.fit(train_data_pca[:,:comp_pca], train_label_pca)
        predict_labels = model_pca.predict(test_data_pca[:,:comp_pca])
    if(indexHighestKlassifier == 2):
        features = "rp"
        transformer = random_projection.GaussianRandomProjection(n_components=comp_rp)
        submission_rp = transformer.fit_transform(submission)

        model_rp = knn(k=k_rp)
        model_rp.fit(train_data_rp[:,:comp_rp], train_label_rp)
        predict_labels = model_rp.predict(test_data_rp[:,:comp_rp])

    Submission = pd.DataFrame({
        "ImageId": range(1, predict_labels.shape[0]+1),
        "Label": predict_labels
    })

    Submission.to_csv("KnnMnistSubmission"+features+".csv", index=False)

    Submission.head(5)

'''
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