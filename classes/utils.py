from classes.knn_lib import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import datetime
import seaborn as sn
import pandas as pd


class Utils():
    def plotKAccuracieDiagram(accuracies,klist, name):
        plt.plot(klist, accuracies)
        plt.title(name)
        plt.ylabel('accuracy')
        plt.xlabel('K')
        plt.savefig('./images/' + name + '.png')
        plt.show()

    def find_k(trainData, trainLabels, testData, testLabels, testSize, description):
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
            model.fit(trainData, trainLabels)
            #print("{} Fitted model with k={}. ".format(datetime.datetime.now(), k))

            # evaluate the model and update the accuracies list
            #print("{} Evaluating model with k={} and a validation data set of size = {}. ".format(datetime.datetime.now(), k, valSize))
            score = model.score(testData[:testSize,:], testLabels[:testSize])
            #print("{} Evaluated model with k={} and a validation data set of size = {}. ".format(datetime.datetime.now(), k, valSize))
            print("{} {}: k={}, accuracy={}%".format(datetime.datetime.now(), description,k, score * 100))
            accuracies.append(score)

        # find the value of k that has the largest accuracy
        i = int(np.argmax(accuracies))
        print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))

        Utils.plotKAccuracieDiagram(accuracies, kVals, description)
        return kVals[i], accuracies[i] * 100


    def plotConfusionMatrix(predictions,testLabels,title):
        #calculate with pandas
        tl = pd.Series((testLabels), name = 'Actual')
        pl = pd.Series((predictions), name = 'Predicted')
        cm = pd.crosstab(tl, pl, rownames=['Actual'], colnames=['Predicted'])
        cm_norm = cm / cm.sum(axis=1) #normalize
        cm_norm.round(2).to_csv('con_mat'+title+'.csv', index=False, header=True)

        #plot
        cm_plt = pd.DataFrame(cm_norm, range(10), range(10))
        sn.set(font_scale=1.0) #for label size
        sn.heatmap(cm_plt.round(2), annot=True, annot_kws={"size" : 8}) #font size
        plt.title('Confusion Matrix' + title + '\n')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show() #save plot as image

    def find_k_with_components(trainData, trainLabels, testData, testLabels, description):
        components = [x for x in range(5, 50, 5)]
        neighbors = [k for k in range(1, 14, 2)]

        scores = np.zeros( (components[len(components)-1]+1, neighbors[len(neighbors)-1]+1 ) )
        for component in range(5, 50, 5):
            for k in range(1, 14, 2):
                # train the k-Nearest Neighbor classifier with the current value of `k`
                # sklearn knn implementation
                #print("{} Initializing model with k={}. ".format(datetime.datetime.now(), k))
                model = KNeighborsClassifier(n_neighbors=k)
                #print("{} Initialized model with k={}. ".format(datetime.datetime.now(), k))
                # our knn implementation
                #model = knn(k=k)
                #print("{} Fitting model with k={}. ".format(datetime.datetime.now(), k))
                model.fit(trainData[:,:component], trainLabels)
                #print("{} Fitted model with k={}. ".format(datetime.datetime.now(), k))

                # evaluate the model and update the accuracies list
                #print("{} Evaluating model with k={} and a validation data set of size = {}. ".format(datetime.datetime.now(), k, valSize))
                score = model.score(testData[:,:component], testLabels)
                #print("{} Evaluated model with k={} and a validation data set of size = {}. ".format(datetime.datetime.now(), k, valSize))
                print("{} {}: k={}, component={}, accuracy={}%".format(datetime.datetime.now(), description, k, component, score * 100))
                scores[component][k] = score

        '''
        # find the value of k that has the largest accuracy
        i = int(np.argmax(accuracies))
        print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[i], accuracies[i] * 100))
        '''
        scores = np.reshape(scores[scores != 0], (len(components), len(neighbors)))
        Utils.plotAccCompKDiagram(scores, neighbors, components, description)
        return Utils.findHighestAccuracie(scores)

    def findHighestAccuracie(scoreMatrix):
        highestAcc = 0
        comp = 0
        k = 0
        for x,component in enumerate(scoreMatrix):
            for y,neighbors in enumerate(component):
                if(scoreMatrix[x][y] > highestAcc):
                    highestAcc = scoreMatrix[x][y]
                    comp = x
                    k = y
        return k*2+1,comp*5+5,highestAcc

    def plotAccCompKDiagram(scoreMatrix, neighbors, components, description):
        plt.rcParams["axes.grid"] = False

        fig, ax = plt.subplots()
        plt.imshow(scoreMatrix, cmap='hot', interpolation='none')
        plt.xlabel('neighbors')
        plt.ylabel('components')
        plt.xticks(range(0,6), neighbors)
        plt.yticks(range(0,9), components)
        plt.title('KNN score heatmap for ' + description)

        plt.colorbar()
        plt.show()


    def getScore(predictions, labels):
        right = 0
        for idx, prediction in enumerate(predictions):
            if(prediction == labels[idx]):
                right += 1
        return right/len(predictions)
