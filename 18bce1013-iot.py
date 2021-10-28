#!/usr/bin/env python
# coding: utf-8

# In[2]:



import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import datetime
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import os


# In[3]:


df = pd.read_csv('heart.csv')
df.shape


# In[4]:


df.head()


# In[5]:


df.target.value_counts()


# In[6]:


df.isna().sum()


# In[7]:


df.describe()


# In[8]:


X = df.drop('target',1)
y = df['target']
print( X.shape, y.shape)


# In[9]:


unique_class, counts_class = np.unique(y, return_counts=True)
fig = plt.figure()
ax = fig.add_axes([0,0,1.5,1.5])
ax.bar(unique_class,counts_class)
ax.set_xlabel('Classes', fontsize='large')
ax.set_ylabel('Count', fontsize='large')

plt.show()


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[11]:


tree_list = [25,50,100,200]


# In[12]:


time_rf_list=[]
rf_accuracy=[]
y_pred_list=[]

for n in tree_list:
    
    rf_clf = RandomForestClassifier(n_estimators=n, random_state=121,criterion='entropy')
       
    rf_clf.fit(X_train, y_train)
   
    y_pred_list.append(rf_clf.predict(X_test))    
      
    rf_accuracy.append(metrics.accuracy_score(y_test,rf_clf.predict(X_test)))


# In[14]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50) 
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))


# In[13]:


rf_accuracy


# In[24]:


import csv


# In[25]:


df = pd.read_csv('proj2.csv')
df


# In[27]:


z=model.predict(df[0:])


# In[28]:


if(z[0]==0):
    str="no heart disease"
else:
    str="heart disease"


# In[31]:


field=['heart disease']
row=[[str]]
filename = "predictor.csv"
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(field) 
        
    # writing the data rows 
    csvwriter.writerows(row)


# In[ ]:




