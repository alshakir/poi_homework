#!/usr/bin/python


#%% cell 1
import sys
import pickle

print 'End of cell 1'


# the following line should be commented if running windows 10
#sys.path.append("../tools/")

#%% cell 2
# the following code is meant for windows10 OS to add the tools directory to the Path
# it should be commented if running Linux
import os
try:
    #This will not work on interactive env. like Jupyter
    dirName = os.path.dirname(__file__)
except NameError:  # We are the main [Jupyter]  script, not a module
    import sys
    dirName = os.path.dirname(os.path.abspath(sys.argv[0]))

#this is for windows10..to be commented in mac
pParent = dirName[:dirName.rindex('\\')+1]

# this is for mac .. to be commented in windows10
pParent = dirName[:dirName.rindex('/')+1]


toolsPath = pParent+'tools'
sys.path.append(toolsPath)
#---------- End of Windows 10 code




print 'End of cell 2'

#%% test cell
print sys.path
print sys.argv
count = 1
for i in data_dict.keys():
    print i, count
    count+=1

print data_dict['TOTAL']['salary']



#%% cell 3 
# this cell is specific for running the code in the REPL interactive like jupyter
#so it should be removed when submitting the project
sys.path =   sys.path[:15]

#these links are for my windows 10 device..to be commented when using mac..
#every user should use the exact location on his device
# tools = "C:\\Users\\alsha\\Dropbox\\DAND_nanodegree\\machineLearning_miniproject\\ud120-projects\\tools"
# testerPath =  "C:\\Users\\alsha\\Dropbox\\DAND_nanodegree\\machineLearning_miniproject\\ud120-projects\\final_project"


# these links are for the mac
tools = '/Users/shakiralharthi/Documents/ud120-projects-master/tools/'
testerPath =  '/Users/shakiralharthi/Documents/ud120-projects-master/final_project/'


sys.path.append(tools)
sys.path.append(testerPath)
count = 0
for i in sys.path:
    print count, '--' ,i
    count +=1

print "end of cell 3"
 
#%% trials and tests
print os.getcwd()

import pprint

pp = pprint.PrettyPrinter(4)

#pp.pprint(data_dict)


import pandas as pd 
import numpy as np 

df = pd.DataFrame.from_dict(data_dict)
print df[:4]["WASAFF GEORGE"]

df.dropna(axis=0,how='any')


print df

#%% cell 4
#from feature_format import featureFormat, targetFeatureSplit

import feature_format
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','total_payments', 'exercised_stock_options',
'bonus', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value',
'expenses', 'loan_advances', 'director_fees', 'deferred_income', 'long_term_incentive'] # You will need to use more features

### Load the dictionary containing the dataset
with open(testerPath + "final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    data_dict.pop('TOTAL')

print 'End of cell 4'



#%% cell 5

reload(feature_format)
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = feature_format.featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = feature_format.targetFeatureSplit(data)

print len(data_dict)
print my_dataset
print
print
print
print '****************'
print 


print data
print len(data)


print len(labels),'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
print labels
print features
print 'End of cell 5'





#%% cell 6

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


clf.fit(features_train,labels_train)

pred = clf.predict(features_test)


from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import recall_score, precision_score


score = accuracy_score(labels_test,pred)

prec_reca_f = precision_recall_fscore_support(labels_test,pred)

print score
print '----------------'

print prec_reca_f
print 'end of cell 6'


print recall_score(labels_test,pred), ' is the recall'
print precision_score(labels_test, pred), 'is the precision'



print labels_test[7]
print pred[7]
counter = 0
wrongcount=0

cc = 1

for i in labels_test:
    if i == 1 or pred[counter]== 1 :
        print 'i = ', i,  ' ___pred = ', pred[counter]
        print cc
        cc+=1
    # if i == pred[counter]:
    #     print i, '****', pred[counter]
    #     print type(i),'type***', type(pred[counter])
    #     print counter
    # else:
    #     wrongcount+=1
    counter+=1

print counter
print 'wrong count = ', wrongcount
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

print 'end of cell 6'

#%% cell 7

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
def useGaussianNB():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()



    clf.fit(features_train,labels_train)

    pred = clf.predict(features_test)


    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.metrics import recall_score, precision_score


    score = accuracy_score(labels_test,pred)

    prec_reca_f = precision_recall_fscore_support(labels_test,pred)

    print score
    print '----------------'

    print prec_reca_f
    print 'end of cell 6'


    print recall_score(labels_test,pred), ' is the recall'
    print precision_score(labels_test, pred), 'is the precision'



    print labels_test[7]
    print pred[7]
    counter = 0
    wrongcount=0

    cc = 1

    for i in labels_test:
        if i == 1 or pred[counter]== 1 :
            print 'i = ', i,  ' ___pred = ', pred[counter]
            print cc
            cc+=1
        # if i == pred[counter]:
        #     print i, '****', pred[counter]
        #     print type(i),'type***', type(pred[counter])
        #     print counter
        # else:
        #     wrongcount+=1
        counter+=1

    print counter
    print 'wrong count = ', wrongcount


print 'end of cell 7'
#%% cell 8.
# svm classifier

def useSVM():
    print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n'
    from sklearn import svm

    clf = svm.SVC()

    clf.fit(features_train,labels_train)

    pred = clf.predict(features_test)


    score = accuracy_score(labels_test,pred)

    prec_reca_f = precision_recall_fscore_support(labels_test,pred)

    print score
    print '----------------'

    print prec_reca_f
    print 'end of cell 6'


    print recall_score(labels_test,pred), ' is the recall'
    print precision_score(labels_test, pred), 'is the precision'




    counter = 0
    wrongcount=0

    cc = 1

    for i in labels_test:
        if i == 1 or pred[counter]== 1 :
            print 'i = ', i,  ' ___pred = ', pred[counter]
            print cc
            cc+=1
        # if i == pred[counter]:
        #     print i, '****', pred[counter]
        #     print type(i),'type***', type(pred[counter])
        #     print counter
        # else:
        #     wrongcount+=1
        counter+=1

    print counter
    print 'wrong count = ', wrongcount

print ' end of cell 8 '



#%% cell 9

def useDTClf():
    from sklearn.tree import DecisionTreeClassifier 

    clf = DecisionTreeClassifier()
    clf.fit(features_train,labels_train)

    pred = clf.predict(features_test)


    score = accuracy_score(labels_test,pred)

    prec_reca_f = precision_recall_fscore_support(labels_test,pred)

    print score
    print '----------------'

    print prec_reca_f
    print 'end of cell 6'


    print recall_score(labels_test,pred), ' is the recall'
    print precision_score(labels_test, pred), 'is the precision'




    counter = 0
    wrongcount=0

    cc = 1

    for i in labels_test:
        if i == 1 or pred[counter]== 1 :
            print 'i = ', i,  ' ___pred = ', pred[counter]
            print cc
            cc+=1
        # if i == pred[counter]:
        #     print i, '****', pred[counter]
        #     print type(i),'type***', type(pred[counter])
        #     print counter
        # else:
        #     wrongcount+=1
        counter+=1

    print counter
    print 'wrong count = ', wrongcount


print 'end of cell 9'
#%% cell testing all classifiers

useGaussianNB()
useSVM()
useDTClf()
print 'end of cell testing all classifiers'