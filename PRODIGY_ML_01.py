#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# In[19]:


data= pd.read_csv("train.csv")


# In[20]:


data


# In[21]:


null_count=data.isnull().sum()
null_count[null_count>0]


# In[22]:


null_count[data.dtypes=='int64']


# In[29]:


data.columns


# In[30]:


from sklearn.decomposition import PCA
PCA_columns = ['TotalBsmtSF','1stFlrSF','2ndFlrSF','GarageArea','PoolArea','GrLivArea','WoodDeckSF','OpenPorchSF','TotRmsAbvGrd']
pca= PCA(n_components=3)
new_cols = pca.fit_transform(data[PCA_columns])
new_cols


# In[31]:


train_X = pd.DataFrame(columns=['Area','Bedrooms','Bathrooms','col1','col2','col3'])
train_X['Area']=data['LotArea']
train_X['Bedrooms']=data['BedroomAbvGr']
train_X['Bathrooms']=data['FullBath'] + data['HalfBath']
train_X['col1'] = new_cols[:, 0]
train_X['col2'] = new_cols[:, 1]
train_X['col3'] = new_cols[:, 2]
train_X


# In[32]:


train_Y = data['SalePrice']
train_Y


# In[33]:


X_tr,X_ts,Y_tr,Y_ts = train_test_split(train_X,train_Y,test_size=0.2,random_state=123)


# In[38]:


print(X_tr.shape)
print(X_ts.shape)
print(Y_tr.shape)
print(Y_ts.shape)


# In[39]:


model = LinearRegression(fit_intercept=False)
model.fit(X_tr, Y_tr)


# In[41]:


model.score(X_ts, Y_ts)


# In[43]:


Y_pred= model.predict(X_ts)
print(r2_score(Y_pred, Y_ts))


# In[59]:


line_creation=LinearRegression()
sample_X = np.array(train_X['col1']).reshape(-1,1)
line_creation.fit(sample_X, train_Y)


# In[60]:


sample_X = np.sort(sample_X)
sample_X[:5]


# In[61]:


plt.scatter(train_X['col1'],train_Y, label='Data')
plt.plot(sample_X,line_creation.predict(sample_X),color='r',label='Model')
plt.xlabel("Feature Vector 1")
plt.ylabel("Price")
plt.title("Regression Model")
plt.legend()
plt.savefig("Regressionmodel.jpg")
plt.show()


# In[ ]:




