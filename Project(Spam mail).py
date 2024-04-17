#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# In[3]:


#importing the data as dataset
dataset = pd.read_csv("E:\DS project\Project Mail\spam_ham_dataset.csv")
dataset


# In[4]:


#processing the step by step
#step 1:EDA
dataset.isnull().sum() # prints if any null values is present or not in dataset


# In[5]:


dataset.info() # prints information about the Dataset


# In[6]:


dataset.describe() # Returns description of the data in the Dataset


# In[7]:


#step 2:Data modelling (training and testind the data )
#split the data from data for testing and training
X = dataset[['Unnamed: 0']]
y = dataset['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[8]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[9]:


LogisticRegression()


# In[10]:


predictions = logmodel.predict(X)
predictions


# In[11]:


predictions = logmodel.predict([[605]])

predictions


# In[12]:


# Evaluate the model on the testing data
accuracy = logmodel.score(X_test, y_test)
print("Accuracy on the testing set:", accuracy)


# In[ ]:




