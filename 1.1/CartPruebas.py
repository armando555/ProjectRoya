# Python implementation of the C4.5 Algorithm
# Uses Pandas dataFrames for managing the dataset
# and can be run by running the main in this file

import pandas as pd
import numpy as np
import random as rnd
import Node
import InputOutput as rg
from time import time
from LinkedList import LinkedList as Lista
from reportlab.pdfgen import canvas
#import graphviz
import pydot
import os
from IPython.display import Image, display
import pydotplus

targetEntropy = 0

# Partition dataSet into training and test partitions
# dataFrame: dataset we want to partition
# test_percentage: percentage of the dataset we want to use for testing
def partitionData(dataFrame, test_percentage):
    tot_index = range(len(dataFrame)) # generate array with values 1 to length of dataset
    test_indexes = rnd.sample(tot_index, int(test_percentage * len(dataFrame))) # get a random sample of indices from this array and by multiplying by % split we want
    train_indexes = list(set(tot_index) ^ set(test_indexes)) # take away the test indexes from the the train indexes using the intersection between the sets total and train

    # use the indexes generated to build up dataframes
    test_df = dataFrame.loc[test_indexes]
    train_df = dataFrame.loc[train_indexes]

    return train_df, test_df


# counts the occurrences of the lables in the dataset
# dataFrameColum: column of the dataset we want to find occurrences for
# returns: results: a dictionary with the class that occurs and the number of times it occurs in the column
def getUniqueClasses(dataFrameColumn):
    results = {}
    for row in dataFrameColumn:
        if row not in results: results[row] = 0 # initialise element in dictionary if not already there
        results[row] += 1 # increment value

    return results


# gets the information entropy for a column in the dataset
# data: dataset
# column: column currently under inverstigations
# returns: entropy: entropy value for the given column
def getEntropy(data, column):
    entropy = 0.0
    results = getUniqueClasses(data[column]) # get # of classes in data

    for row in results.values():
        p = float(row) / len(data[column]) # calculat probability value for each element in results
        entropy -= p * np.log2(p) # calculate entropy

    return entropy


# Finds the thresholds(splitpoints) in a dataset for a given column
# data: dataset
# column: column we want to find thresholds for
# returns: splitpoints: a list of indexes in the dataframe where splitpoints occur
#
# This method splits the data where a change in the decision class occurs
# rather than splitting at every row (brute force method)
# This makes it more efficient as it generates less splitpoints for the data
def findSplitPoints(data, column):
    sorted = data.sort_values([column], ascending=True)
    sorted_matrix = sorted[[column, 'label']].as_matrix()
    splitPoints = []
    previous = sorted_matrix[0][1] # get target of the first first element in sorted matrix
    index = sorted.index.values # get the indexes of each  element in the sorted matrix
    counter = 0
    for row in sorted_matrix:
        if row[1] != previous: # if the type (class) of row 1 = the previous type
            splitPoints.append([index[counter - 1], sorted_matrix[counter - 1][0]])
        counter += 1
        previous = row[1]

    return splitPoints # indexes of the dataframe where splits should occur


# Splits the dataset into subsets based on thresholds
# data: dataset to find subsets for
# column: column of the dataset we are currently subsetting
# splitpoints: an index of thresholds to use when subsetting the data
# returns: sets_below: a list of dataframes that contain data below the given splitpoints
#          sets_above: a list of dataframes that contain data above the given splitpoints
def splitSets(data, column, splitPoints):
    sets_below = []
    sets_above = []
    # split the dataframe into 2 for each splitpoint
    for i in range(len(splitPoints)):
        df1 = data[data[column] <= data[column][splitPoints[i][0]]]  # everything below the splitpoint
        df2 = data[data[column] > data[column][splitPoints[i][0]]]  # everything above it
        # add to the lists
        sets_below.append(df1)
        sets_above.append(df2)

    return sets_below, sets_above


# Gets the information gain of splitting the data on a given splitpoint in the data
# data: dataframe containing the data
# column: column of dataset that we want to find the best IG for
# returns: best_gain: best information gain for the given colum
#          sets_below: the subset of the data that is below the splitpoint that gives the best IG
#          sets_above: the subset of the data that is above the splitpoint that gives the best IG
#          splitpoints: the splitpoint that gives the best IG
def getInformationGain(data, column):
    splitpoints = findSplitPoints(data, column)  # get splitpoints for this column
    sets_below, sets_above = splitSets(data, column, splitpoints)  # split the data into sets based on these splitpoints
    # lists to store the # of instances in each subset that are above and below each given threshold and their entropies
    instances_above = []
    instances_below = []
    entropy_above = []
    entropy_below = []
    target_entropy = getEntropy(data, 'label')  # get target entropy for the dataset
    # get entropy for sets above and below each of the thresholds
    for set in sets_below:
        entropy_below.append(getEntropy(set, 'label'))
        instances_below.append(len(set))
    for set in sets_above:
        entropy_above.append(getEntropy(set, 'label'))
        instances_above.append(len(set))

    totalInstances = []
    infoGains = []
    # work out the Information Gain for each threshold
    for i in range(len(instances_below)):
        totalInstances.append(instances_below[i] + instances_above[i])
        probA = (instances_above[i] / float(totalInstances[i]))
        probB = (instances_below[i] / float(totalInstances[i]))
        infoGains.append(target_entropy - ((entropy_below[i] * probB) + (entropy_above[i] * probA)))

    # work out the highest information gain for this column of the dataset
    best_gain = i = counter = 0
    for gain in infoGains:
        if best_gain < gain:
            best_gain = gain
            counter = i # variable to hold the index in the list where the best gain occurs
        i += 1

    return best_gain, sets_below[counter], sets_above[counter], splitpoints[counter]


# Build the tree
# data: dataframe that is going to be used to build the tree.
# Must be of the format:
# col1, col2, col3, ...coln, type where type is the decision class
def train(data):
    optimal_gain = -1
    best = {}
    columns = []
    i = 0

    for column in data:  # loop over each attribute
        if column != 'label':
            try:
                ig, set1, set2, split = getInformationGain(data, column)  # get information gain for each column
                # column holds information that is used when creating a tree node.
                # the values in each of the columns will be used below when creating nodes for the tree
                columns.append({"ig": ig, "left": set1, "right": set2, 'col': i, 'split': split,
                                'colName': column})  # append attributes to list to generate tree node

            # above code will work until the set1 and set2 values that would be returned bu the informationGain function will be 0
            # in that case, an indexError will be thrown as we cannot access element 0 of the sets lists
            # so if we catch that exception we know this data should be used as a leaf node and can format the tree information accordingly
            except IndexError:
                columns.append({"ig": 0, "left": [], "right": [], 'col': column, })
        i += 1  # counter to get int value for row(used for tree node)

    # loops through each column and pulls out the one with the best information gain for the given data
    for val in range(len(columns)):

        if columns[val]['ig'] > optimal_gain:
            best = columns[val]
            optimal_gain = columns[val]['ig']

    # get data for left branch and data for right branch
    left = best['left']
    right = best['right']
    # check if we have data for the left and right branches of the tree
    # if they are = 0 it is the stop condition for recursion, and the else block will generate a leaf node for the tree
    if len(best['left']) != 0 and len(best['right']) != 0:
        return (Node.treeNode(col=best['col'], colName=best['colName'], value=best['split'][1], results=None,
                              rb=train(right), lb=train(left)))

    else:
        label = list(getUniqueClasses(data['label']).keys()) #get label for the leaf node
        return (Node.treeNode(results=label[0]))


# Traverse the tree to find the leaf node that the target_row will be classified as
# modelled after the print function below written by Conor
# target_row: row of a dataframe that we are classifying
# tree: decision tree
def classify(target_row, tree):
    # base case to stop recursion -> we are at a leaf node
    if tree.results != None:
        return tree.results

    else:
        # gets the attribute from the target row that we are looking at
        val = target_row[tree.col]
        branch = None
        if isinstance(val, int) or isinstance(val, float):
            # checks the value of the tree against the value of the attribute from the target row
            # go down right side
            if val >= tree.value:
                branch = tree.rb
            # go down left side
            else:
                branch = tree.lb
        # recur over the tree again, using either the left or right branch to determine where to go next
        return classify(target_row, branch)


# Prints the tree in a readable format
# tree: decision tree we want to print
# space: spacing we want to use for child nodes
def printTree(tree, space=''):
    # Leaf node
    if tree.results != None:
        print(space+str(tree.results))

    else:
        print(space+str(tree.colName) + ' : ' + str(tree.value)) #name and splitpoint of current node
        print(space + '|-->Left:¬') #Print 'L' for left child
        printTree(tree.lb, space + '|'+" ") #Print left child
        print(space + '└─>Right:¬ ') #Print 'R' for right child
        printTree(tree.rb, space + ''+" ") #Print right child


rootDir = ""
def exportTree(tree, node, g):
    #leaf Node
    if (tree.results != None):
       node_temp = pydot.Node(str(tree.results),shape="plaintext")
       g.add_node(node_temp)
       edge = pydot.Edge(str(node),str(node_temp))
       g.add_edge(edge)

    else:
       node_temp = pydot.Node((str(tree.colName)+' : '+str(tree.value)), style="filled",fillcolor="green")
       g.add_node(node_temp)
       edge = pydot.Edge(str(node),str(node_temp))
       g.add_edge(edge)
       exportTree(tree.lb, node_temp, g)
       exportTree(tree.rb, node_temp, g)


# Tests the tree and returns how accurate the classifier is
# data: test data we want to classify
# labels: list of labels that we want to cross reference the classifier result with (stripped out of test_data before it is passed in)
# tree: decision tree we want to use for testing
# returns: % accuracy for the tree
def test_tree(data, labels, tree):
    values = []
    # Loop over each row in test data frame and get the classification result for each index
    for index, row in data.iterrows():
        values.append([index, classify(row, tree)])

    # Get the indexes from the test dataframe where each type occurs
    indexes = labels.index.values
    correct = incorrect = 0
    # Loop over values list and compare the class that was classified by the tree
    # and the class that was originally in the dataframe
    for l in range(len(values)):
        if values[l][0] == indexes[l] and values[l][1] == labels[indexes[l]]:
            correct += 1 #increment the correctly classified #
        else:
            incorrect += 1 #increment the incorrectly classified #

    return incorrect, correct, np.round(100 - (incorrect / (incorrect + correct)) * 100)


def main():
    # Read dataSet from file into a dataframe
    # Any dataset can be used, as long as the last column is the result
    # And the columns have headings, with the last column called 'type'
    dataSet = pd.read_csv('data_set.csv')
    # dataSet = pd.read_csv('datasets/illness.csv')

    results = []
    tests = 10 # number of times the algorithm will be run (more runs will give a more accurate average accuracy)
    # loop to test the tree. Each loop it:
    # -> generates random data partitions
    # -> generates a decision tree
    # -> classifies the test_data using this decision tree
    # -> gets the accuracy of the decision tree
    # -> gets the average accuracy, over all the iterations
    for i in range(tests):
        train_data, test_data = partitionData(dataSet, 0.9) # random partitions
        tree = train(train_data) # make tree
        types = test_data['label'] # get types column from test_data
        del test_data['label'] # deletes labels from test_data so it cannot be used in classification

        incorrect, correct, accuracy = test_tree(test_data, types, tree) # test the tree
        results.append(accuracy)

        # print information to console
        print("Test " + str(i + 1) + "\n------------")
        print("Tree Generated:" + "\n")
        printTree(tree)
        
        #print("Tree.lb "+str(tree.lb))
        #print("Tree.lb "+str(tree.rb))
        #begin PDF EXPORT
        g=pydot.Dot(graph_type="digraph")
        node = pydot.Node(str(tree.value), style="filled", fillcolor="green")
        g.add_node(node)
        exportTree(tree, node, g)
        #im = Image(g.create_png())
        #display(im)
        g.write_pdf("tree.pdf")
        
        #end PDF EXPORT
        print()
        print("Correctly Classified: " + str(correct) + " / " + str(correct+incorrect))
        print("Accuracy: " + str(accuracy))
        print()

    # get the average accuracy for all the runs
    sum = 0
    for r in range(len(results)):
        sum += results[r]
    average = sum/tests

    print("Average Accuracy after " + str(tests) + " runs")
    print(average)


#--------------------------------------------------------------------------------------MEMORY IN MB--------------------------------
def memoryUsage():
    # return the memory usage in MB
    import psutil
    import os 
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)
    return mem

if __name__ == '__main__':
    tiempoInicial=time()
    main()
    tiempoFinal=time()
    tiempo=tiempoFinal-tiempoInicial
    features=[]
    labels=[]
    direccion='data_set.csv'
    '''
    el ciclo toma cada linea hace un split de ella y la lleva a una lista que se agrega
    a la lista features, pero antes le quita la etiqueta "yes" o "no" para poder
    emplear el machine learning después, y esta etiqueta se guarda en labels
    '''
    # la función leerArchivo fue creada por mi para verificar que el archivo existe e abrirlo
    with rg.leerArchivo(direccion) as archivo:
        for linea in archivo:
            linea=linea.split(",")
            if linea[6]=="yes\n":
                linea[6]=1
            else:
                linea[6]=0
            labels.append(linea.pop(6))
            features.append(linea)

    #-------------------------------------------Linked List--------------------------------
    lista=Lista()
    count=0
    timeAddI=time()
    for feature in features:
        listaN=feature
        listaN.append(labels[count])
        lista.add(listaN)
        count=count+1
    timeAddF=time()
    timeAdd=timeAddF-timeAddI
    timeImprimirI=time()
    lista.print()
    timeImprimirF=time()
    tiempoImprimir=timeImprimirF-timeImprimirI
    print()
    print()
    timeSearchI=time()
    print(lista.search(features[50]).data)
    timeSearchF=time()
    timeSearch=timeSearchF-timeSearchI
    timeDeleteI=time()
    lista.delete(features[0])
    timeDeleteF=time()
    timeDelete=timeDeleteF-timeDeleteI

    timeReplaceS=time()
    replaces=[6.23,27.0,19.2,1204.0,36.0,42.0,"yes"]
    lista.replace(replaces, 0)
    timeReplaceE=time()
    timeReplace=timeReplaceE-timeReplaceS

    memoria=memoryUsage()
    print("------------------------------------------------ aca hice delete")
    lista.print()
    #ESTE ES EL TIEMPO DE EJECUCIÓN DEL DECISION TREE
    print()
    print("TIEMPO DE EJECUCION DEL DECISION TREE")
    print(tiempo)
    #ESTE ES EL TIEMPO DE EJECUCIÓN DEL PRINT LIST
    print()
    print("TIEMPO DE EJECUCION DEL PRINT LIST: ")
    print(tiempoImprimir)
    #ESTE ES EL TIEMPO DE EJECUCIÓN DEL ADD LIST
    print()
    print("TIEMPO DE EJECUCION DEL ADD LIST: ")
    print(timeAdd)
    #ESTE ES EL TIEMPO DE EJECUCIÓN DEL SEARCH LIST
    print()
    print("TIEMPO DE EJECUCION DEL SEARCH LIST: ")
    print(timeSearch)
    #ESTE ES EL TIEMPO DE EJECUCIÓN DEL DELETE LIST
    print()
    print("TIEMPO DE EJECUCION DEL DELETE LIST: ")
    print(timeDelete)
    #ESTE ES EL TIEMPO DE EJECUCIÓN DEL REPLACE IN LIST
    print()
    print("TIEMPO DE EJECUCION DEL REPLACE LIST: ")
    print(timeReplace)
    #MEMORY IN MB
    print()
    print("MEMORIA EN MB: \n",memoria)
    print()
    #PROMEDIO DE LOS TIEMPOS
    promedio=(timeDelete+timeSearch+tiempoImprimir+timeAdd)/4
    print("EL PROMEDIO DE LOS TIEMPOS ES: \n",promedio)