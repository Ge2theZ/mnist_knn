from classes.knn_lib import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import datetime
import seaborn as sn

class Utils():
    def plotKAccuracieDiagram(accuracies,klist, name):
        plt.plot(klist, accuracies)
        plt.title(name)
        plt.ylabel('accuracy')
        plt.xlabel('K')
        plt.savefig('./images/kAccuracies_k1_37800samples_sklearn.png')
        plt.show()

    def find_k(train, val, labelsTrain, labelsVal, trainSize, valSize, description):
        # initialize the values of k for our k-Nearest Neighbor classifier along with the
        # list of accuracies for each value of k
        kVals = range(1, 30, 2)
        accuracies = []
        # loop over various values of `k` for the k-Nearest Neighbor classifier
        for k in range(1, 30, 2):
            # train the k-Nearest Neighbor classifier with the current value of `k`
            # sklearn knn implementation
            #print("{} Initializing model with k={}. ".format(datetime.datetime.now(), k))
            model = KNeighborsClassifier(n_neighbors=k)
            #print("{} Initialized model with k={}. ".format(datetime.datetime.now(), k))
            # our knn implementation
            #model = knn(k=k)
            #print("{} Fitting model with k={}. ".format(datetime.datetime.now(), k))
            model.fit(train[:trainSize, :], labelsTrain[:trainSize])
            #print("{} Fitted model with k={}. ".format(datetime.datetime.now(), k))

            # evaluate the model and update the accuracies list
            #print("{} Evaluating model with k={} and a validation data set of size = {}. ".format(datetime.datetime.now(), k, valSize))
            score = model.score(val[:valSize, :], labelsVal[:valSize])
            #print("{} Evaluated model with k={} and a validation data set of size = {}. ".format(datetime.datetime.now(), k, valSize))
            print("{} {}: k={}, accuracy={}%".format(datetime.datetime.now(), description,k, score * 100))
            accuracies.append(score)

        # find the value of k that has the largest accuracy
        i = int(np.argmax(accuracies))
        print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i],
                                                                               accuracies[i] * 100))
        Utils.plotKAccuracieDiagram(accuracies,kVals, description)
        return kVals[i], accuracies[i] * 100

    def plotConfusionMatrix(predictions,testLabels):
        #calculate with pandas
        tl = pd.Series((testLabels), name = 'Actual')
        pl = pd.Series((predictions), name = 'Predicted')
        cm = pd.crosstab(tl, pl, rownames=['Actual'], colnames=['Predicted'])
        cm_norm = cm / cm.sum(axis=1) #normalize
        cm_norm.round(2).to_csv('con_mat.csv', index=False, header=True)

        #plot
        cm_plt = pd.DataFrame(cm_norm, range(10), range(10))
        sn.set(font_scale=1.4) #for label size
        sn.heatmap(cm_plt.round(2), annot=True, annot_kws={"size" : 16}) #font size
        plt.title('Confusion Matrix \n')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show() #save plot as image
