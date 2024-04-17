#!/usr/bin/env python
# coding: utf-8

# In[27]:


#Importing the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


# In[28]:


#importing the data as dataset
dataset = pd.read_csv("E:\DS project\Project Mail\spam_ham_dataset.csv")
dataset


# In[29]:


#processing the step by step
#step 1:EDA
dataset.isnull().sum() # prints if any null values is present or not in dataset


# In[30]:


dataset.info() # prints information about the Dataset


# In[31]:


dataset.describe() # Returns description of the data in the Dataset


# In[21]:


#step 2:Data modelling (training and testind the data )
#split the data from data for testing and training
X = dataset[['Unnamed: 0']]
y = dataset['label_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[22]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[23]:


LogisticRegression()


# In[24]:


predictions = logmodel.predict(X)
predictions


# In[25]:


predictions = logmodel.predict([[605]])

predictions


# In[ ]:




