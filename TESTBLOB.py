#!/usr/bin/env python
# coding: utf-8

# In[2]:


# pip install -U textblob


# In[7]:


from textblob import TextBlob
from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger


# In[8]:


b = TextBlob("I havv goood speling!")
print(b.correct())


# In[ ]:




