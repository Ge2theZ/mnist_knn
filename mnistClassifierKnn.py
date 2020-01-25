"""
Source: https://gurus.pyimagesearch.com/lesson-sample-k-nearest-neighbor-classification/#
Comment: Made some minor adjustments
"""

# import the necessary packages
from __future__ import print_function
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import numpy as np
import imutils
import cv2
import sklearn
import pandas as pd

# read the mnist train data
mnistData = pd.read_csv("data/train.csv")
mnistTest = pd.read_csv("data/test.csv")
dfMnist = pd.DataFrame(mnistData)
dfTest = pd.DataFrame(mnistTest)

# Split the mnist train data into a dataframe of train and validation labels
dfVal = pd.DataFrame(mnistData).iloc[int(mnistData.shape[0] * 0.9):, :]
dfTrain = pd.DataFrame(mnistData).iloc[:int(mnistData.shape[0] * 0.9), :]

print("dfTrain Size: ", dfTrain.shape[0])
print("dfVal Size: ", dfVal.shape[0])

# Convert the train dataframe to an array of features and labels
labelsTrainArr = dfTrain['label'].to_numpy()
trainArr = dfTrain.loc[:, 'pixel0':].to_numpy()

print("labelsTrainArr Size: ", labelsTrainArr.size)
print("trainArr Elements: ", trainArr.shape[0])

# Convert the val dataframe to an array of features and labels
labelsValArr = dfVal['label'].to_numpy()
valArr = dfVal.loc[:, 'pixel0':].to_numpy()

print("labelsValArr Size: ", labelsValArr.size)
print("valArr Elements: ", valArr.shape[0])

# Convert the test dataframe to an array of features
testArr = dfTest.loc[:, 'pixel0':].to_numpy()
print("testArr Elements: ", testArr.shape[0])

# We splitted our train data into an array of train data and an array of val data. The train data is used to train our
# classifier. The val array is used to retrieve the best value for k and to retrieve a score for our classifier.
# The test data is data which does not contain any labels. This is the actual data that needs to be classified and
# submitted in the kaggle contest.


# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []

trainSize = 1000
valSize = 100

# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 30, 2):
    # train the k-Nearest Neighbor classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainArr[:trainSize, :], labelsTrainArr[:trainSize])

    # evaluate the model and update the accuracies list
    score = model.score(valArr[:valSize,:], labelsValArr[:valSize])
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                       accuracies[i] * 100))

# re-train our classifier using the best k value and predict the labels of the
# test data
model = KNeighborsClassifier(n_neighbors=kVals[i])
model.fit(trainArr[:trainSize, :], labelsTrainArr[:trainSize])
predictions = model.predict(valArr[:valSize,:])

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(labelsValArr[:valSize], predictions))

# loop over a few random digits
for i in list(map(int, np.random.randint(0, high=valSize, size=(5,)))):
    # grab the image and classify it
    image = valArr[i]
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
