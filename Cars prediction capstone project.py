#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('CAR DETAILS.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.isnull().sum()


# In[6]:


df['fuel'].value_counts()


# In[7]:


df['seller_type'].value_counts()


# In[8]:


df['transmission'].value_counts()


# In[9]:


df['owner'].value_counts()


# In[10]:


df.describe()


# In[11]:


df.info()


# In[12]:


df['name'].unique()


# ### Performing EDA on the dataset

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


plt.scatter(df['name'],df['selling_price'])


# ### Performing data preprocessing

# In[15]:


df1=df.copy()
df1.head()
df1['name'].value_counts()


# In[16]:


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
enc_list = ['fuel','seller_type','transmission','owner']
for i in enc_list:
    df1[i] = lb.fit_transform(df1[i])


# In[48]:


df1.head()


# ### Data cleaning

# In[18]:


df1['name'] = df['name'].apply(lambda x:" ".join(x.split()[0:2]))


# In[19]:


df1.head(5)


# In[20]:


df1.to_csv('cleaned df1.csv')


# ### Model building

# In[21]:


x = df1.drop(columns=['name','selling_price'])
y = df1['selling_price']


# In[22]:


print(x)


# In[23]:


# spliting the dataset into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)


# In[24]:


# importing the liberaries
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn import metrics


# In[25]:


lr=LinearRegression()


# In[26]:


lr.fit(x_train,y_train)


# In[27]:


# prediction on training data
training_df1_prediction = lr.predict(x_train)


# In[28]:


error_score = metrics.r2_score(y_train,training_df1_prediction)
print('R squared Error:',error_score)


# In[29]:


plt.scatter(y_train,training_df1_prediction)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Actual price vs Predicted price')
plt.show()


# In[30]:


# prediction on testing data
testing_df1_prediction = lr.predict(x_test)
error_score = metrics.r2_score(y_test,testing_df1_prediction)
print('R squared Error',error_score)


# In[32]:


Lass_rm = Lasso()
Lass_rm.fit(x_train,y_train)


# In[33]:


training_df1_prediction = Lass_rm.predict(x_train)
error_score = metrics.r2_score(y_train,training_df1_prediction)
print('R Squared Error:',error_score)


# In[35]:


testing_df1_prediction = Lass_rm.predict(x_test)
error_score = metrics.r2_score(y_test,testing_df1_prediction)
print('R Squared Error:',error_score)


# In[37]:


# decision tree regressor
from sklearn.tree import DecisionTreeRegressor


# In[38]:


dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)


# In[39]:


testing_df1_prediction = dtr.predict(x_test)
error_score = metrics.r2_score(y_test,testing_df1_prediction)
print('R Squared Error:',error_score)


# In[49]:


training_df1_prediction = dtr.predict(x_train)
error_score = metrics.r2_score(y_train,training_df1_prediction)
print('R Squared Error:',error_score)


# In[42]:


plt.scatter(y_train,training_df1_prediction)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Actual price vs Predicted price')
plt.show()


# In[43]:


from sklearn.ensemble import RandomForestRegressor


# In[44]:


rfr = RandomForestRegressor()
rfr.fit(x_train,y_train)


# In[45]:


testing_df1_prediction = rfr.predict(x_test)
error_score = metrics.r2_score(y_test,testing_df1_prediction)
print('R Squared Error:',error_score)


# In[46]:


plt.scatter(y_train,training_df1_prediction)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Actual price vs Predicted price')
plt.show()


# In[47]:


training_df1_prediction = rfr.predict(x_train)
error_score = metrics.r2_score(y_train,training_df1_prediction)
print('R Squared Error:',error_score)


# ### Inference:- Based on R squared error decisiontreeregressor performs better as compared to other model.

# In[51]:


import pickle
pickle.dump(df1,open('df1.pkl','wb'))
pickle.dump(dtr,open('dtr.pkl','wb'))


# In[ ]:




