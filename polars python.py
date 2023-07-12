#!/usr/bin/env python
# coding: utf-8

# In[12]:


#pip install polars


# In[3]:


import polars as pl


# In[16]:


df=pd.read_csv("C:\\Users\\HP\\Desktop\Fl.csv")
print(df.head())


# In[20]:


df.tail()


# In[8]:


df.describe()


# In[10]:


df.isnull().sum()


# In[18]:


df.dtypes


# In[ ]:




