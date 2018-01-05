#!/usr/bin/python



import sys
import pickle


''' there is a problem of directory locating and reaching to script files...

There are 3 scenarios (Linux, Mac, and windows 10)

------- Secnario one (linux) ----------
sys.path.append("../tools/")

------- End of Secnario one (linux) ----------



------- Secanrio two and three (Win10 and Mac OS) ----------

# the following code is meant for windows10 OS and MacOS to add the 'tools directory' to the Path
# it should be commented if running Linux
import os
try:
    #This will not work on interactive env. like Jupyter
    dirName = os.path.dirname(__file__)
except NameError:  # We are the main [Jupyter]  script, not a module
    import sys
    dirName = os.path.dirname(os.path.abspath(sys.argv[0]))

# this is for windows10..to be commented in mac
pParent = dirName[:dirName.rindex('\\')+1]

# this is for mac .. to be commented in windows10
pParent = dirName[:dirName.rindex('/')+1]

------- End of Secanrio two and three (Win10 and Mac OS) ----------




toolsPath = pParent+'tools'
finalProjPath = pParent + 'final_project'
sys.path.append(toolsPath)
sys.path.append(finalProjPath)

------to test that the  paths are correct print them ------
# print dirName
# print pParent
# print toolsPath
# print finalProjPath



************ HOWEVER ALL THE ABOVE COMMENTED CODE IS SUBSTITUTED BY PUTTING THE FILES OF THE TOOL
DIRECTORY ON THE SAME DIRECTORY OF THIS CODE, AND BY THIS WE ELIMINATE THE NEED 
FOR DIGGING UP THE DIRECTORY TREES **************

'''




import feature_format
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.

features_list = ['poi','from_poi_percentage', 'to_poi_percentage','salary',
'bonus','exercised_stock_options', 'deferred_income'] 



### Load the dictionary containing the dataset
with open( "final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### Task 2: Remove outliers
    data_dict.pop('TOTAL')







### Task 3: Create new feature(s)
### we will create two new features that are the percentage of the total email that the person send to or recieve from POI:
#           'to_poi_percentage',
#           'from_poi_percentage'


my_dataset = data_dict

#---------- adding new features----------


for i in my_dataset.keys():
    x = float( my_dataset[i]['to_messages'])
    y = float(my_dataset[i]['from_messages'])
    to_poi= float( my_dataset[i]['from_this_person_to_poi'])
    
    from_poi = float(my_dataset[i]['from_poi_to_this_person'])
    totalMessages = x + y


    my_dataset[i]['to_poi_percentage'] = 0
    my_dataset[i]['from_poi_percentage'] = 0
    if totalMessages > 0:
        my_dataset[i]['to_poi_percentage'] = to_poi/totalMessages
        my_dataset[i]['from_poi_percentage'] = from_poi/totalMessages

#----------End of adding new features----------

### Extract features and labels from dataset for local testing
data = feature_format.featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = feature_format.targetFeatureSplit(data)




### Task 4: Try a varity of classifiers


### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.25, random_state=42)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

print 'end of cell 6'



### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.






#%%
print features_list

#%% cell 11
# testing gaussianNB with kfold


# print "This is the useGaussianNBKfold() method"
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.cross_validation import cross_val_score

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import recall_score, precision_score
import numpy as np


kf = KFold(n_splits=7,shuffle=True)
clf = GaussianNB()
# print type(features)

# print '-----------\n\n\n'

# print cross_val_score(clf,features,labels,scoring=None,cv=7)
# print '-----------\n\n\n'
# print '-----------\n\n\n'



dump_classifier_and_data(clf, my_dataset, features_list)


f = np.array(features)
l = np.array(labels)
# print f.shape
# print l.shape

recList = []
presList = []

for train_index, test_index in kf.split(features):
    # print train_index
    # print '====='
    # print test_index
    # print'*********\n\n'
    features_train = f[train_index]
    labels_train = l[train_index]
    features_test = f[test_index]
    labels_test = l[test_index]


    clf.fit(features_train,labels_train)

    pred = clf.predict(features_test)


    score = accuracy_score(labels_test,pred)

    prec_reca_f = precision_recall_fscore_support(labels_test,pred)

    # print score
    # print '----------------'

    # print prec_reca_f
    # print 'end of cell 6'

    rec =  recall_score(labels_test,pred)
    pres = precision_score(labels_test, pred)
    print rec, ' is the recall'
    print pres, 'is the precision'


    recList.append(rec)
    presList.append(pres)


    # print labels_test[7]
    # print pred[7]
    counter = 0
    wrongcount=0

    cc = 1

    # for i in labels_test:
    #     if i == 1 or pred[counter]== 1 :
    #         print 'i = ', i,  ' ___pred = ', pred[counter]
    #         print cc
    #         cc+=1
    #     # if i == pred[counter]:
    #     #     print i, '****', pred[counter]
    #     #     print type(i),'type***', type(pred[counter])
    #     #     print counter
    #     # else:
    #     #     wrongcount+=1
    #     counter+=1

    # print counter
    # print 'wrong count = ', wrongcount
avgRecall = np.average(recList)
avgPrecision = np.average(presList)


print "average recall =  ", avgRecall
print "average precision  = " , avgPrecision


# useGaussianNBKfold()
print 'end of cell 11'