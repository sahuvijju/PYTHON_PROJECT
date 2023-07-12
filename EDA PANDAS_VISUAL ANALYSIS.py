#!/usr/bin/env python
# coding: utf-8

# # EDA USING PANDAS_VISUALIZE

# In[1]:


#pip install pandas_visual_analysis


# In[2]:


import seaborn as sns


# In[3]:


print(sns.get_dataset_names())


# In[4]:


df=sns.load_dataset('tips')
print(df)


# In[5]:


df.head()


# In[6]:


from pandas_visual_analysis import VisualAnalysis


# In[7]:


df.describe()


# In[8]:


df.tail()


# In[9]:


df.info()


# In[10]:


df.isnull().sum()


# In[11]:


VisualAnalysis(df)


# In[12]:


import matplotlib.pyplot as plt
plt.scatter(df['tip'],df['sex'])


# In[ ]:





# In[ ]:





# In[ ]:




