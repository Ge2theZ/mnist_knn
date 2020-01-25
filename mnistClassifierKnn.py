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
from knn_lib import *


# handle older versions of sklearn
if int((sklearn.__version__).split(".")[1]) < 18:
    from sklearn.cross_validation import train_test_split

# otherwise we're using at lease version 0.18
else:
    from sklearn.model_selection import train_test_split

# load the MNIST digits dataset
mnist = datasets.load_digits()

# take the MNIST data and construct the training and testing split, using 75% of the
# data for training and 25% for testing
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data),
                                                                  mnist.target, test_size=0.25, random_state=42)

# now, let's take 10% of the training data and use that for validation
(trainData, valData, trainLabels, valLabels) = train_test_split(trainData, trainLabels,
                                                                test_size=0.1, random_state=84)

# show the sizes of each data split
print("original sklearn MNIST dataset length: {}".format(len(mnist.data)))
print("training data points: {}".format(len(trainLabels)))
print("validation data points: {}".format(len(valLabels)))
print("testing data points: {}".format(len(testLabels)))
print("How the data looks like: \n", trainData)

# initialize the values of k for our k-Nearest Neighbor classifier along with the
# list of accuracies for each value of k
kVals = range(1, 30, 2)
accuracies = []

# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 30, 2):
    # train the k-Nearest Neighbor classifier with the current value of `k`
    #our knn implementation
    model = knn(k=k)
    #sklearn knn implementation
    #model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)

    # evaluate the model and update the accuracies list
    score = model.score(valData, valLabels)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                       accuracies[i] * 100))

# re-train our classifier using the best k value and predict the labels of the
# test data
#model = KNeighborsClassifier(n_neighbors=kVals[i])
model = knn(k=kVals[i])
model.fit(trainData, trainLabels)
predictions = model.predict(testData)

# show a final classification report demonstrating the accuracy of the classifier
# for each of the digits
print("EVALUATION ON TESTING DATA")
print(classification_report(testLabels, predictions))

# loop over a few random digits
for i in list(map(int, np.random.randint(0, high=len(testLabels), size=(5,)))):
    # grab the image and classify it
    image = testData[i]
    prediction = model.predict(image.reshape(1, -1))[0]

    # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
    # then resize it to 32 x 32 pixels so we can see it better
    image = image.reshape((8, 8)).astype("uint8")
    image = exposure.rescale_intensity(image, out_range=(0, 255))
    image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    # show the prediction
    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", image)
    cv2.waitKey(0)

#------------------------------- Confusion Matrix -----------------------------------------

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

#calculate with pandas
tl = pd.Series((testLabels), name = 'Actual')
pl = pd.Series((predictions), name = 'Predicted')
cm = pd.crosstab(tl, pl, rownames=['Actual'], colnames=['Predicted'])
cm_norm = cm / cm.sum(axis=1) #normalize
print(cm_norm.round(2))
cm_norm.round(2).to_csv('con_mat.csv', index=False, header=True)

#plot
cm_plt = pd.DataFrame(cm_norm, range(10), range(10))
sn.set(font_scale=1.4) #for label size
sn.heatmap(cm_plt.round(2), annot=True, annot_kws={"size" : 16}) #font size
plt.title('Confusion Matrix \n')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show() #save plot as image


