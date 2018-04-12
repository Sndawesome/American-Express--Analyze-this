# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 11:27:51 2017

@author: Sandesh
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import sklearn.tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectPercentile, f_classif

train=pd.read_csv("F:/Analytics/Analyze/Training_Dataset.csv")
test=pd.read_csv("F:/Analytics/Analyze/Final_Dataset.csv")
dataset=[test,train]
num=preprocessing.LabelEncoder()

train.drop(['mvar46','mvar47','mvar48'],axis=1,inplace=True)

clf=GaussianNB()
train['mvar50'].replace(train.mvar50[train.mvar50==1],2,inplace=True)
train['mvar51'].replace(train.mvar51[train.mvar51==1],3,inplace=True)
train['mvar52']=train.mvar49+train.mvar51+train.mvar50
for data in dataset:
   data.mvar16=data.mvar16+data.mvar17+data.mvar18+data.mvar19
   data.drop(['mvar18','mvar17','mvar19'],axis=1,inplace=True)
   data.mvar20=data.mvar20+data.mvar21+data.mvar22+data.mvar23
   data.drop(['mvar21','mvar22','mvar23'],axis=1,inplace=True)
   data.mvar24=data.mvar24+data.mvar25+data.mvar26+data.mvar27
   data.drop(['mvar25','mvar26','mvar27'],axis=1,inplace=True)
   data.mvar28=data.mvar28+data.mvar29+data.mvar30+data.mvar31
   data.drop(['mvar29','mvar30','mvar31'],axis=1,inplace=True)
   data.mvar32=data.mvar32+data.mvar33+data.mvar34+data.mvar35
   data.drop(['mvar33','mvar34','mvar35'],axis=1,inplace=True)
   data.mvar36=data.mvar36+data.mvar37+data.mvar38+data.mvar39
   data.drop(['mvar38','mvar37','mvar39'],axis=1,inplace=True)
   data['mvar12']=num.fit_transform(data['mvar12'].astype(str))
# transformimg data   
X=train.iloc[:,2:].values
X = scale(X)
pca = PCA(n_components=30)
pca.fit(X)
var= pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)


predictors=['mvar2','mvar11','mvar13','mvar14','mvar15','mvar40', 'mvar41', 'mvar42', 'mvar43', 'mvar44', 'mvar45']
x_train=train[predictors].values
x_test=test[predictors].values 
clf.fit(x_train,train['mvar52'])
z=clf.predict_proba(x_test)
y_test=clf.predict(x_test)
acc=clf.score(x_test,y_test)
test['mvar52']=y_test
'''plt.scatter(train.index[train.mvar52==0],train.mvar36[train.mvar52==0],color='r')
plt.scatter(train.index[train.mvar52==1],train.mvar36[train.mvar52==1],color='b')
plt.scatter(train.index[train.mvar52==2],train.mvar36[train.mvar52==2],color='g')
plt.scatter(train.index[train.mvar52==3],train.mvar36[train.mvar52==3],color='pink')
plt.show()'''
#scores = cross_val_score(clf, x_test,cv=4)
test['mvar52'] = test['mvar52'].map( {1: 'Supp', 2: 'Elite', 3: 'Credit'} ).astype(str) 
test1=test.drop(test.index[test.mvar52=='nan'])
test1[:1000].to_csv('Pheonix_IITG.csv',columns=['cm_key','mvar52'],index=0,header=False)
sub=pd.read_csv('Pheonix_IITG.csv')
# random forest
from sklearn.ensemble import RandomForestClassifier
clf1=RandomForestClassifier(n_estimators=25 )

clf1.fit(x_train,train['mvar52'])
p=clf1.predict(x_test)
test['mvar53']=p
acc1=clf1.score(x_test,p)
test['mvar53'] = test['mvar53'].map( {1: 'Supp', 2: 'Elite', 3: 'Credit'} ).astype(str)
test2=test.drop(test.index[test.mvar53=='nan']) 
test2[:1000].to_csv('Pheonix_IITG_4.csv',columns=['cm_key','mvar53'],index=0,header=False)
from sklearn.neural_network import MLPClassifier
clf2=MLPClassifier(solver='adam', alpha=1e-5, random_state=1)

clf2.fit(x_train,train['mvar52'])
p=clf2.predict(x_test)
test['mvar54']=p
test['mvar54'] = test['mvar54'].map( {1: 'Supp', 2: 'Elite', 3: 'Credit'} ).astype(str)
test3=test.drop(test.index[test.mvar54=='nan']) 
test3[:1000].to_csv('Pheonix_IITG_3.csv',columns=['cm_key','mvar53'],index=0,header=False)
