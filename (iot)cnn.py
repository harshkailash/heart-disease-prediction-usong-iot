#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#matplotlib inline

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')


# In[3]:


#Importing and understanding our dataset
dataset = pd.read_csv("heart.csv")


# In[4]:


#Verifying it as a 'dataframe' object in pandas
type(dataset)


# In[5]:


#Shape of dataset
dataset.shape


# In[6]:


#Printing out a few columns
dataset.head(5)


# In[7]:


dataset.sample(5)


# In[8]:


#Description
dataset.describe()


# In[9]:


dataset.info()


# In[10]:


#understanding the columns better:
info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(dataset.columns[i]+":\t\t\t"+info[i])


# In[11]:


#Analysing the 'target' variable
dataset["target"].describe()


# In[12]:


dataset["target"].unique()


# In[13]:


#Checking correlation between columns
print(dataset.corr()["target"].abs().sort_values(ascending=False))
#This shows that most columns are moderately correlated with target, but 'fbs' is very weakly correlated.


# In[14]:


#Exploratory Data Analysis (EDA)
#First, analysing the target variable:
y = dataset["target"]

sns.countplot(y)


target_temp = dataset.target.value_counts()

print(target_temp)


# In[15]:


print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))


# In[16]:


dataset["sex"].unique()


# In[17]:


#We notice, that as expected, the 'sex' feature has 2 unique features
sns.barplot(dataset["sex"],y)


# In[18]:


#Analysing the 'Chest Pain Type' feature
dataset["cp"].unique()


# In[19]:


#As expected, the CP feature has values from 0 to 3
sns.barplot(dataset["cp"],y)
#We notice, that chest pain of '0', i.e. the ones with typical angina are much less likely to have heart problems


# In[20]:


#Analysing the FBS feature
dataset["fbs"].describe()


# In[21]:


dataset["fbs"].unique()


# In[22]:


sns.barplot(dataset["fbs"],y)


# In[23]:


#Analysing the restecg feature
dataset["restecg"].unique()


# In[24]:


sns.barplot(dataset["restecg"],y)
#We realize that people with restecg '1' and '0' are much more likely to have a heart disease than with restecg '2'


# In[25]:


#Analysing the 'exang' feature
dataset["exang"].unique()


# In[26]:


sns.barplot(dataset["exang"],y)
#People with exang=1 i.e. Exercise induced angina are much less likely to have heart problems


# In[27]:


#Analysing the Slope feature
dataset["slope"].unique()


# In[28]:


sns.barplot(dataset["slope"],y)
#We observe, that Slope '2' causes heart pain much more than Slope '0' and '1'


# In[29]:


#Analysing the 'ca' feature
#number of major vessels (0-3) colored by flourosopy
dataset["ca"].unique()


# In[30]:


sns.countplot(dataset["ca"])


# In[31]:


sns.barplot(dataset["ca"],y)
#ca=4 has astonishingly large number of heart patients


# In[32]:


#Analysing the 'thal' feature
dataset["thal"].unique()


# In[33]:


sns.barplot(dataset["thal"],y)


# In[34]:


sns.distplot(dataset["thal"])


# In[35]:


#Train Test split
from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)


# In[36]:


X_train.shape


# In[37]:


X_test.shape


# In[38]:


Y_train.shape


# In[39]:


Y_test.shape


# In[40]:


#Model Fitting
from sklearn.metrics import accuracy_score


# In[41]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)
Y_pred_lr.shape


# In[42]:


score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")


# In[43]:


# #Naive Bayes
# from sklearn.naive_bayes import GaussianNB

# nb = GaussianNB()

# nb.fit(X_train,Y_train)

# Y_pred_nb = nb.predict(X_test)

# Y_pred_nb.shape


# In[44]:


# score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

# print("The accuracy score achieved using Naive Bayes is: "+str(score_nb)+" %")


# In[45]:


#SVM
from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)
Y_pred_svm.shape


# In[46]:


score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)
print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")


# In[47]:


#K Nearest Neighbors
# from sklearn.neighbors import KNeighborsClassifier

# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(X_train,Y_train)
# Y_pred_knn=knn.predict(X_test)
# Y_pred_knn.shape


# In[48]:


# score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

# print("The accuracy score achieved using KNN is: "+str(score_knn)+" %")


# In[49]:


# #Decision Tree
# from sklearn.tree import DecisionTreeClassifier

# max_accuracy = 0


# for x in range(200):
#     dt = DecisionTreeClassifier(random_state=x)
#     dt.fit(X_train,Y_train)
#     Y_pred_dt = dt.predict(X_test)
#     current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
#     if(current_accuracy>max_accuracy):
#         max_accuracy = current_accuracy
#         best_x = x
        
# #print(max_accuracy)
# #print(best_x)


# dt = DecisionTreeClassifier(random_state=best_x)
# dt.fit(X_train,Y_train)
# Y_pred_dt = dt.predict(X_test)
# print(Y_pred_dt.shape)


# In[50]:


# score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

# print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")


# In[51]:


# #XGBoost
# import xgboost as xgb

# xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
# xgb_model.fit(X_train, Y_train)

# Y_pred_xgb = xgb_model.predict(X_test)
# Y_pred_xgb.shape


# In[52]:


# score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)

# print("The accuracy score achieved using XGBoost is: "+str(score_xgb)+" %")


# In[53]:


#Neural Network
from keras.models import Sequential
from keras.layers import Dense


# In[54]:


model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[55]:


model.fit(X_train,Y_train,epochs=300)


# In[56]:


Y_pred_nn = model.predict(X_test)


# In[57]:


Y_pred_nn.shape


# In[58]:


rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded


# In[59]:


score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

print("The accuracy score achieved using Neural Network is: "+str(score_nn)+" %")

#Note: Accuracy of 85% can be achieved on the test set, by setting epochs=2000, and number of nodes = 11.


# In[62]:


#Output final score
scores = [score_lr,score_svm,score_nn]
algorithms = ["Logistic Regression","Naive Bayes","Neural Network"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(scores[i])+" %")


# In[63]:


sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,scores)


# In[ ]:




