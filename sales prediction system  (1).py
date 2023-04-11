#!/usr/bin/env python
# coding: utf-8

# # importing packages

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# # reading csv file

# In[4]:


data = pd.read_csv("C:\\Users\\SANTH\\OneDrive\\Documents\\Advertising.csv")
print(data)


# # using model

# In[ ]:


x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)


# # displaying the predicted data

# In[3]:


data = pd.DataFrame(data={"Predicted Sales": ypred.flatten()})
print(data)


# In[ ]:





# In[ ]:




