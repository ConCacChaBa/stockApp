#!/usr/bin/env python
# coding: utf-8

# # Stock prediction

# ## Part 1. Download and describe the data

# ### Load the data

# In[5]:


import numpy as np
import pandas as pd
data = pd.read_csv(r'C:\Users\phatm\Desktop\Project Ngoc Dung\BGFV.csv')
data.head()


# In[6]:


data = data.drop(['Date'], axis=1)
print(data)


# ## Predition model by using Support Vector Machines (SVM)

# ### Classifier's learning
# ##### Initial classifier: SVC class, polynomial degree 3 kernel and gamma = 0.001

# In[7]:


label = data.take([-1], axis=1)
data = data.drop(['Predicted'], axis=1)
data = data.to_numpy()
label = label.to_numpy()
data = data[:-1]
label = label[:-1]*100


n_samples = len(label)
slipt = int(n_samples*2/3)
di = data[:slipt]
dt = label[:slipt]
ti = data[slipt:]
tt = label[slipt:]

dt = dt.ravel().astype('int')
tt = tt.ravel().astype('int')
# from sklearn import preprocessing
# lab_enc = preprocessing.LabelEncoder()
# dt = lab_enc.fit_transform(dt)
# tt = lab_enc.fit_transform(tt)

print(dt)

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import timeit

start = timeit.default_timer()
print('\n'+'\033[1m')

# clf = svm.SVC(kernel='poly', degree = 2,gamma = 0.001)
clf = RandomForestClassifier(n_estimators = 100, min_samples_split=2)

print('0')

clf.fit(di,dt)
print('1')

predicted = clf.predict(ti)
print('2 \nDone!')
print('\033[0m')

stop = timeit.default_timer()
print('Run time: ', stop - start)

# ### Classifier's prediction

# In[10]:


c, d = predicted, tt
import re
print(re.sub(r' *\n *', '\n', np.array_str(np.c_[d, c]).replace('[', '').replace(']', '').strip()))


# ### Classifier's report, confusion matrix and cross-validation's score


from sklearn import metrics
from sklearn.model_selection import cross_val_score

print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(tt, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(tt, predicted))

print("\n\nEvaluate a score by cross-validation:     mean score = %s \n" % cross_val_score(clf, ti, tt, cv=3).mean())
