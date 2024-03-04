# -------------------------------------------------------------------------
# AUTHOR: Ahmad Alkadi
# FILENAME: decision_tree
# SPECIFICATION: train and test by using different training sets which goes around 10 times and then averaging the accuracies as the final classification performance of each model
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 30 min
# -----------------------------------------------------------*/

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    for row in data_training:
        features = []
        features.append(1 if row[0] == 'Yes' else 2)
        features.append(1 if row[1] == 'Single' else 0)
        features.append(1 if row[1] == 'Divorced' else 0)
        features.append(1 if row[1] == 'Married' else 0)
        features.append(int(float(row[2].rstrip('k'))))
        X.append(features)

    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    for row in data_training:
        Y.append(1 if row[3] == 'Yes' else 2)


    #loop your training and test tasks 10 times here
    for i in range (10):

       #fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)

       #plotting the decision tree
       tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
       plt.show()

       #read the test data and add this data to data_test NumPy
       #--> add your Python code here
       data_test = df = pd.read_csv('cheat_test.csv', sep=',', header=0)
       data_test = np.array(data_test.values)[:, 1:]

       cheatCount = 0
       for data in data_test:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here
           featuresTwo = []
           featuresTwo.append(1 if data[0] == 'Yes' else 2)
           featuresTwo.append(1 if data[1] == 'Single' else 0)
           featuresTwo.append(1 if data[1] == 'Divorced' else 0)
           featuresTwo.append(1 if data[1] == 'Married' else 0)
           featuresTwo.append(int(float(data[2].rstrip('k'))))
           class_predicted = clf.predict([featuresTwo])[0]


           #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
           #--> add your Python code here
           label = (1 if data[3] == 'Yes' else 2)
           #print(label)
           if class_predicted == label:
               cheatCount += 1



       #find the average accuracy of this model during the 10 runs (training and test set)
       #--> add your Python code here
       accuracy = cheatCount / len(data_test)

    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    #--> add your Python code here
    print(f'final accuracy when training on{ds}: {accuracy}')



