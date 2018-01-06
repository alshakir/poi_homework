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
### we will create two new features that are the percentage of the total email that the person sent to or recieved from POI:
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

def useDTClf():
    
    print "This is the useDTClf() method"
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier 

    #clf = DecisionTreeClassifier(max_depth=1)

    param = {'max_depth': [1,2,3,4,5,6,7,8,9],
    'min_samples_split':[2,3,4,5,6,7,8,9],
    'min_samples_leaf':[1,2,3,4,5,6,7,8,9]}
    from sklearn.model_selection import GridSearchCV
    clf = GridSearchCV(DecisionTreeClassifier(), param)


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



    print 'best params = ', clf.best_params_
    print 'best estimator = ', clf.best_estimator_
    print 'best score = ', clf.best_score_
    print 'best index = ', clf.best_index_
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

    # import graphviz
    # dot_data = tree.export_graphviz(clf, out_file=None) 
    # graph = graphviz.Source(dot_data) 
    # graph.render("poi") 

def useSVM():
    
    print "This is the useSVM() method"
    print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n'
    from sklearn import svm

    #clf = svm.SVC()

    param = {'kernel': ['linear', 'rbf','poly', 'rbf', 'sigmoid'] ,
    'C':[1,2,3,4,5,6,7,8,9,10, 300,5000],
    'decision_function_shape' : ['ovo', 'ovr']
    }
    from sklearn.model_selection import GridSearchCV
    clf = GridSearchCV(svm.SVC(max_iter=8000000), param)

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




    print 'best params = ', clf.best_params_
    print 'best estimator = ', clf.best_estimator_
    print 'best score = ', clf.best_score_
    print 'best index = ', clf.best_index_

def usePCAKnearst():
    print 'this is the usePCAKnearest method'
    from sklearn.decomposition import PCA

    myPCA = PCA(n_components=3)
    transformed_pca_features_train= myPCA.fit_transform(features_train)

    transformed_pca_features_test = myPCA.fit_transform(features_test)
    from sklearn.neighbors import KNeighborsClassifier
    # clf = KNeighborsClassifier()
    # clf.fit(transformed_pca_features_train,labels_train)

    # pred = clf.predict(transformed_pca_features_test)


    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    
    params = {'pca__n_components':[2,3],
    'knearest__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}


    

    pipe = Pipeline(steps=[('pca', PCA()), ('knearest', KNeighborsClassifier())])
    clf = GridSearchCV(pipe, params )


    clf.fit(features_train,labels_train)

    pred = clf.predict(features_test)



    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.metrics import recall_score, precision_score



    print recall_score(labels_test,pred), ' is the recall'
    print precision_score(labels_test, pred), 'is the precision'

def usePCA_SVM():
    
    print 'this is pca svm piping and gridsearch'
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn import svm

    params = {'pca__n_components':[2,3],
    'svm__kernel': ['linear', 'rbf'] ,
    'svm__C':[1,2,3,100,1000]}


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    scaled_features_train = scaler.fit_transform(features_train)
    scaled_features_test = scaler.fit_transform(features_test)

    print 'datamax of scaler = ', scaler.data_max_

    print scaled_features_train[:3]
    
    pipe = Pipeline(steps=[('pca', PCA()), ('svm', svm.SVC(max_iter=1000000))])
    clf = GridSearchCV(pipe, params )



    clf.fit(scaled_features_train,labels_train)

    pred = clf.predict(scaled_features_test)





    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.metrics import recall_score, precision_score



    print recall_score(labels_test,pred), ' is the recall'
    print precision_score(labels_test, pred), 'is the precision'

def usePCA_DTC():
    
    print ' this is PCA with DTClf no piping'
    from sklearn.decomposition import PCA

    myPCA = PCA(n_components=3)
    transformed_pca_features_train= myPCA.fit_transform(features_train)

    transformed_pca_features_test = myPCA.fit_transform(features_test)
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    clf.fit(transformed_pca_features_train,labels_train)

    pred = clf.predict(transformed_pca_features_test)





    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.metrics import recall_score, precision_score



    print recall_score(labels_test,pred), ' is the recall'
    print precision_score(labels_test, pred), 'is the precision'





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


import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold,StratifiedKFold

from sklearn.cross_validation import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score
import numpy as np


kf = KFold(n_splits=7,shuffle=True)
clf = GaussianNB()

f = np.array(features)
l = np.array(labels)

recallList = []
precisionList = []

for train_index, test_index in kf.split(features):
    features_train = f[train_index]
    labels_train = l[train_index]
    features_test = f[test_index]
    labels_test = l[test_index]

    clf.fit(features_train,labels_train)

    pred = clf.predict(features_test)

    score = accuracy_score(labels_test,pred)

    recall =  recall_score(labels_test,pred)
    precision = precision_score(labels_test, pred)
    print '**** New Fold ****'
    print recall, ' is the recall'
    print precision, 'is the precision'
    print 



    recallList.append(recall)
    precisionList.append(precision)


avgRecall = np.average(recallList)
avgPrecision = np.average(precisionList)

print '=============Final Result=================='
print "average recall =  ", avgRecall
print "average precision  = " , avgPrecision







'''
Example run  of this code

**** New Fold ****
0.4  is the recall
1.0 is the precision

**** New Fold ****
1.0  is the recall
0.5 is the precision

**** New Fold ****
0.333333333333  is the recall
1.0 is the precision

**** New Fold ****
0.333333333333  is the recall
0.5 is the precision

**** New Fold ****
0.5  is the recall
0.333333333333 is the precision

=============Final Result==================
average recall =   0.366666666667
average precision  =  0.47619047619






example result of the tester.py
GaussianNB(priors=None)
	Accuracy: 0.85707	Precision: 0.49964	Recall: 0.34350	F1: 0.40711	F2: 0.36640
	Total predictions: 14000	True positives:  687	False positives:  688	False negatives: 1313	True negatives: 11312


'''






### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)



