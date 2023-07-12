#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install evalml 


# In[2]:


#pip install -r core-requirements.txt


# In[3]:


#pip install evalml[complete]


# In[4]:


import evalml


# In[5]:


X, y = evalml.demos.load_breast_cancer()
X_train, X_test, y_train, y_test = evalml.preprocessing.split_data(X, y, problem_type='binary')


# In[6]:


X


# In[7]:


X_train.head()


# # Running the automl to select best algorithmn

# In[8]:


import evalml 
evalml.problem_types.ProblemTypes.all_problem_types


# In[9]:


from evalml import AutoMLSearch
automl=AutoMLSearch(X_train=X_train,
    y_train=y_train,
    problem_type="binary",
    objective="f1",
    max_batches=3,
    verbose=True,)
automl.search()


# In[10]:


automl.rankings


# In[11]:


automl.best_pipeline


# In[13]:


best_pipeline = automl.best_pipeline


# # let's check the detailed description

# In[14]:


automl.describe_pipeline(automl.rankings.iloc[0]['id'])


# # evaluate on the out hold data

# In[15]:


best_pipeline.score(X_test,y_test,objectives=["auc","f1","Precision","Recall"])


# In[23]:


automl_auc = AutoMLSearch(
    X_train=X_train,
    y_train=y_train,
    problem_type="binary",
    objective="auc",
    additional_objectives=[ "f1", "precision"],
    
    max_batches=1,
    optimize_thresholds=True
    
)

automl_auc.search()


# In[24]:


automl_auc.rankings


# In[26]:


best_pipeline_auc = automl_auc.best_pipeline


# In[27]:


best_pipeline_auc.score(X_test,y_test,objectives=["auc"])


# In[28]:


automl_auc.describe_pipeline(automl_auc.rankings.iloc[0]['id'])


# In[29]:


best_pipeline_auc = automl_auc.best_pipeline


# In[30]:


best_pipeline_auc.score(X_test,y_test,objectives=["auc"])


# In[32]:


best_pipeline.save("model.pkl")


# # loading the model

# In[52]:


import pandas as pd
check_model=automl.load('model.pkl')


# In[59]:


proba_arr=check_model.predict_proba(X_test)
proba_df=pd.DataFrame(proba_arr)
print(proba_df)


# In[ ]:




