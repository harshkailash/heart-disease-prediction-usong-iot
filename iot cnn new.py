#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import packages that we will be working with.
import os
import numpy as np
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# In[1]:


# Load the dataset, and view couple of the first rows.
data = pd.read_csv("heart1.csv")
print(data.head(3))

# Check the datatypes
print(data.dtypes)


# In[4]:


# At this moment we have a dataframe that contains all of the heart.csv data. However we need to
# Separate them to [X, Y]. Where our target labels are 'Y', and 'X' is our training data.
Y = data.target.values
X = data.drop(['target'], axis=1)

# Now split to train/test with 80% training data, and 20% test data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Check dimensions of both sets.
print("Train Features Size:", X_train.shape)
print("Test Features Size:", X_test.shape)
print("Train Labels Size:", Y_train.shape)
print("Test Labels Size:", Y_test.shape)


# In[5]:


# Neural Network Model
# Lets create a function that we can call later that builds our Neural Network model, and takes in the learning rate as a parameter. The architecture of the Neural Network that we're going to implement is detailed in the below illustration.
# In our model, we use Adam (Adaptive Momentum) as our optimization algorithm, and set our metrics to accuracy. Furthermore, I have used the loss function to be sparse_categorical_crossentropy because our traget labels are integers and not one hot encoded.
# Define a Neural Network Model

def NN_model(learning_rate):
    model = Sequential()
    model.add(Dense(32, input_dim=5, kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, activation='softmax'))
    Adam(lr=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


# In[6]:


# Now lets build the NN-model and start training. I chose learning_rate=0.01, epochs=100, and batch_size=16.
# Training the model for 100 epochs, seems to be pretty fine in order to avoid overfitting. I already performed training with 1000 epochs and around 100 epochs was the reasonable number of epochs for early stopping.
# Lets take a look at our model summary.
# Build a NN-model, and start training

learning_rate = 0.01
model = NN_model(learning_rate)
print(model.summary())


# In[7]:


history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=200, batch_size=16, verbose=2)


# In[8]:


# We plot the Model Accuracy, and Model Loss vs. the number of Epochs.
# Plot the model accuracy vs. number of Epochs

# Plot the Loss function vs. number of Epochs
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'])
plt.show()


# In[9]:


# We compute our model's predictions on the test set X_test and print a classfication_report from the imported package sklearn.metrics.
predictions = np.argmax(model.predict(X_test), axis=1)
model_accuracy = accuracy_score(Y_test, predictions)*100
print("Model Accracy:", model_accuracy,"%")
print(classification_report(Y_test, predictions))


# In[ ]:




