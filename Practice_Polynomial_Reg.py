#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[: , 1:-1].values
y = dataset.iloc[: , -1].values
print(dataset)


# In[4]:


## Linear Regression model on the whole data set


# In[5]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


# In[6]:


## Polynomial  Regression model on the whole data set


# In[7]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit( x_poly, y)


# In[8]:


## Visualize Linear Regression model


# In[9]:


plt.scatter(x,y,color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('visualization of linear regressio model')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


# In[10]:


## Visualize polynomial Regression model


# In[11]:


plt.scatter(x,y,color = 'red')
plt.plot(x, lin_reg_2.predict(x_poly), color = 'blue')
plt.title('visualization of polynomial regressio model')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()


# In[12]:


## Predicting a new result with linear Regression


# In[16]:


y_pred = lin_reg.predict([[4.5]])
y_pred


# In[ ]:


## Predicting a new result with Polynomial Regression


# In[18]:


lin_reg_2.predict(poly_reg.fit_transform([[12]]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




