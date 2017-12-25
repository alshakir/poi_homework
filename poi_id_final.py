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
#pParent = dirName[:dirName.rindex('\\')+1]

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
# print os.getcwd()
# import pprint
# pp = pprint.PrettyPrinter(4)
# #pp.pprint(data_dict)
# import pandas as pd 
# import numpy as np 

# df = pd.DataFrame.from_dict(data_dict)
# print df[:4]["WASAFF GEORGE"]

# df.dropna(axis=0,how='any')

# print df

import numpy as np 
from sklearn.model_selection import StratifiedKFold

X = np.ones(10)
print X
y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=3)
for train, test in skf.split(X, y):
    print 'train: ', train
    print 'test: ', test
    print("%s %s" % (train, test))
    print '----'

#%% cell 4
#from feature_format import featureFormat, targetFeatureSplit

features_list = None


def useFeaturesMain():
    import feature_format
    from tester import dump_classifier_and_data

    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    global features_list
    features_list = ['poi','from_poi_percentage','to_poi_percentage','salary','total_payments', 'exercised_stock_options',
    'bonus', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value',
    'expenses', 'loan_advances', 'director_fees', 'deferred_income', 'long_term_incentive'] # You will need to use more features

    ### Load the dictionary containing the dataset
    with open(testerPath + "final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
        data_dict.pop('TOTAL')



def useFeatures3():
    import feature_format
    from tester import dump_classifier_and_data

    ### Task 1: Select what features you'll use.
    ### features_list is a list of strings, each of which is a feature name.
    ### The first feature must be "poi".
    global features_list
    features_list = ['poi','bonus', 'deferred_income','restricted_stock'] # You will need to use more features

    ### Load the dictionary containing the dataset
    with open(testerPath + "final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
        data_dict.pop('TOTAL')

    


#useFeaturesMain()
useFeatures3()
print features_list
print 'End of cell 4'



#%% cell 5

reload(feature_format)
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
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

#----------------------------------------

### Extract features and labels from dataset for local testing
data = feature_format.featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = feature_format.targetFeatureSplit(data)

# print 'length of data_dict: ',len(data_dict)
# for i in my_dataset['METTS MARK'].keys():
#     print i
#     print my_dataset['METTS MARK'][i]
# print
# print my_dataset['METTS MARK']
# print
# print '****************'
# print 




#print data
# print 'length of data',len(data)


# print 'length of labels : ', len(labels),'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^'
# print 'now following is lables: ......\n\n\n'
# print labels

# print 'length of features :' , len(features)
# print 'Now following is features: ...\n\n\n'
# print features
# print 'End of cell 5'





#%% cell 6

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

print 'This is gaussianNB without using KFold'

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
    print "This is the useGaussianNB() method with no kfold"
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    print clf
    print features_train
    print labels_train
    
    clf.fit(features_train,labels_train)

    pred = clf.predict(features_test)


    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from sklearn.metrics import recall_score, precision_score


    score = accuracy_score(labels_test,pred)

    prec_reca_f = precision_recall_fscore_support(labels_test,pred)

    print score
    print '----------------'

    print prec_reca_f
    


    print recall_score(labels_test,pred), ' is the recall'
    print precision_score(labels_test, pred), 'is the precision'



    # print labels_test[7]
    # print pred[7]
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
useGaussianNB()

print 'end of cell 7'
# #%% cell 8.
# # svm classifier

# def useSVM():
    
#     print "This is the useSVM() method with gridsearchCV"
#     print '^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n'
#     from sklearn import svm

#     #clf = svm.SVC()

#     param = {'kernel': ['linear', 'rbf','poly', 'rbf', 'sigmoid'] ,
#     'C':[1,2,3],
#     'decision_function_shape' : ['ovo', 'ovr']
#     }
#     from sklearn.model_selection import GridSearchCV
#     clf = GridSearchCV(svm.SVC(max_iter=8000000), param)

#     clf.fit(features_train,labels_train)

#     pred = clf.predict(features_test)


#     score = accuracy_score(labels_test,pred)

#     prec_reca_f = precision_recall_fscore_support(labels_test,pred)

#     print score
#     print '----------------'

#     print prec_reca_f
#     print 'end of cell 6'


#     print recall_score(labels_test,pred), ' is the recall'
#     print precision_score(labels_test, pred), 'is the precision'




#     counter = 0
#     wrongcount=0

#     cc = 1

#     for i in labels_test:
#         if i == 1 or pred[counter]== 1 :
#             print 'i = ', i,  ' ___pred = ', pred[counter]
#             print cc
#             cc+=1
#         # if i == pred[counter]:
#         #     print i, '****', pred[counter]
#         #     print type(i),'type***', type(pred[counter])
#         #     print counter
#         # else:
#         #     wrongcount+=1
#         counter+=1

#     print counter
#     print 'wrong count = ', wrongcount




#     print 'best params = ', clf.best_params_
#     print 'best estimator = ', clf.best_estimator_
#     print 'best score = ', clf.best_score_
#     print 'best index = ', clf.best_index_
# useSVM()
# print ' end of cell 8 '



#%% cell 9

def useDTClf():
    
    print "This is the useDTClf() method with gridsearchCV"
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

useDTClf()
print 'end of cell 9'
#%% cell testing all classifiers

useGaussianNB()

#useSVM()
useDTClf()
print 'end of cell testing all classifiers'


#%%

useGaussianNB()


#%%
print features_list

#%% cell 11
# testing gaussianNB with kfold

def useGaussianNBKfold(labels,features):
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
        # print rec, ' is the recall'
        # print pres, 'is the precision'
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

    findTheBest(avgRecall,avgPrecision)
    # print "average recall =  ", avgRecall
    # print "average precision  = " , avgPrecision

    
# useGaussianNBKfold()
print 'end of cell 11'

listBest =[[],0,1]
def findTheBest(rec,prec):
    import numpy as np
    sumRecPrec = rec + prec
    diffRecPrec = np.abs(rec - prec)
    global listBest
    theSum = listBest[1]
    theDiff = listBest[2]

    if(sumRecPrec > theSum and diffRecPrec < theDiff):
        listBest[0] = thefinalList
        listBest[1] = sumRecPrec
        listBest[2] = diffRecPrec
        print listBest
        print 'recall = ',rec, ' And precision = ', prec
      

    

#%% cell12
def usePCA():
    
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








usePCA()
print 'end of cell 12'




#%% cell 13

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

usePCA_SVM()
print 'end of cell 13'





#%% cell14
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








usePCAKnearst()
print 'end of cell 14'




#%% cell 18
thefinalList = ['poi']
def combination(l):
    ggg = 0
    

    global thefinalList 
    thefinalList = ['poi']
    theLength= len(l)
    print (theLength)

    for i in range(theLength):
        
        
        thefinalList.append(l[i])
        for j in range(i+1,theLength):
            thefinalList.append(l[j])
            for k in range(j+1,theLength):
                thefinalList.append(l[k])

                # print thefinalList
                ggg = ggg + 1
                labels,features = loadFeaturesAndLabels(thefinalList)
                useGaussianNBKfold(labels,features)
                #do the job
                thefinalList.pop()
            thefinalList.pop()
        thefinalList.pop()
    print ggg
    global listBest
    listBest =[[],0,1]



myList = ['from_poi_percentage','to_poi_percentage','salary','total_payments', 'exercised_stock_options',
    'bonus', 'restricted_stock', 'restricted_stock_deferred', 'total_stock_value',
    'expenses', 'loan_advances', 'director_fees', 'deferred_income', 'long_term_incentive'] # You will need to use more features


combination(myList)
 
print 'end of cell 18'




#%%

def loadFeaturesAndLabels(features_list):

    reload(feature_format)
    ### Task 2: Remove outliers
    ### Task 3: Create new feature(s)
    ### Store to my_dataset for easy export below.
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

    #----------------------------------------

    ### Extract features and labels from dataset for local testing
    data = feature_format.featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = feature_format.targetFeatureSplit(data)

    return labels, features





#%% test3

lis = [[],0,1]
print lis

lis[0]= ['sdf','sdf']

print lis
print 'end of test3 cell'