#!/usr/bin/env python
# coding: utf-8

# In[44]:


#Importing the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# In[33]:


#importing the data as dataset
dataset = pd.read_csv("E:\DS project\Project Mail\spam_ham_dataset.csv")
dataset


# In[34]:


#processing the step by step
#step 1:EDA
dataset.isnull().sum() # prints if any null values is present or not in dataset


# In[35]:


dataset.info() # prints information about the Dataset


# In[36]:


dataset.describe() # Returns description of the data in the Dataset


# In[37]:


# Train and test Dataset
X_train,X_test,y_train,y_test=train_test_split(dataset.text,dataset.label_num,test_size=0.25)


# In[39]:


clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])


# In[40]:


#Train the Machine
clf.fit(X_train,y_train)


# In[47]:


#Getting a mail for prediction
emails=['Sounds great! Are you home now?']
clf.predict(emails)


# In[43]:


# Prediction Of Model
clf.score(X_test,y_test)


# In[50]:


#ploting
sns.countplot(x='label', data=dataset)
plt.title('Distribution of Ham and Spam Emails')
plt.show()


# In[ ]:




