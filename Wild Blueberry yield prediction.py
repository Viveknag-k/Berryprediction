#!/usr/bin/env python
# coding: utf-8

# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#             IMPORTING REQUIRED LIBRARIES
# </p>
# </div>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)


# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#             IMPORTING DATA
# </p>
# </div>

# In[2]:


df = pd.read_csv("C:\\Users\\Vivek Nag Kanuri\\Downloads\\WildBlueberryPollinationSimulationData.csv")
df.head()


# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#             EXPLORATORY ANALYSIS
# </p>
# </div>

# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.describe()


# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#             FRUIT MASS AND SEEDS
# </p>
# </div>

# In[6]:


plt.figure(figsize=(7,6))
sns.lmplot(x='fruitmass',y='seeds',data=df)
plt.show()


# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#             AVERAGE RAINING DAYS vs YIELD
# </p>
# </div>

# In[7]:


plt.figure(figsize=(7,6))
sns.violinplot(x='AverageRainingDays',y='yield',data=df)
plt.show()


# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#            YIELD W.R.T RAINING DAYS
# </p>
# </div>

# In[8]:


plt.figure(figsize=(7,6))
sns.violinplot(x='RainingDays',y='yield',data=df)
plt.show()


# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#             AVERAGE TEMPERATURE vs YIELD
# </p>
# </div>

# In[9]:


px.histogram(df,df['AverageOfUpperTRange'],df['yield'],template='ggplot2')


# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#             YIELD OF BLUEBERRIES W.R.T CLONESIZE OF BERRIES
# </p>
# </div>

# In[10]:


px.histogram(df, df['clonesize'], df['yield'],template='ggplot2',color='clonesize')


# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#             YIELD OF BLUEBERRIES W.R.T HONEYBEES
# </p>
# </div>

# In[11]:


px.pie(df, df['honeybee'],df['yield'],template='ggplot2',hole=0.6)


# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#             YIELD OF BLUEBERRIES W.R.T BUMBLE BEES
# </p>
# </div>

# In[12]:


px.pie(df, df['bumbles'],df['yield'],template='ggplot2',hole=0.6,color='bumbles')


# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#             YIELD OF BLUEBERRIES W.R.T ANDRENA BEES
# </p>
# </div>

# In[13]:


px.pie(df, df['andrena'],df['yield'],template='ggplot2',hole=0.6,color='andrena')


# <div style="color:black;
#            display:fill;
#            border-radius:30px;
#            background-color:#9DFC8E;
#            font-size:110%;
#            font-family:Verdana;
#            letter-spacing:1px">
# 
# <p style="padding: 10px;
#               color:black;">
#             YIELD OF BLUEBERRIES W.R.T OSMIA BEES
# </p>
# </div>

# In[14]:


px.pie(df, df['osmia'],df['yield'],template='ggplot2',hole=0.6,color='osmia')


# In[15]:


df1 = df.copy(deep=True)


# In[16]:


df1.head()


# In[17]:


df1.drop(['Row#','MaxOfUpperTRange','MinOfUpperTRange','MaxOfLowerTRange','MinOfLowerTRange','RainingDays','fruitset'],axis=1,inplace=True)


# In[18]:


df1.head()


# In[19]:


from sklearn.model_selection import train_test_split
X = df1.drop('yield',axis=1)
y = df1['yield']

xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.30,random_state=42)


# In[20]:


from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)

from sklearn import metrics
print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(ytest, ypred),3))  
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(ytest, ypred),3))  
print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(ytest, ypred)),3))
print('R2_score:', round(metrics.r2_score(ytest, ypred),6))
print('Root Mean Squared Log Error (RMSLE):', round(np.log(np.sqrt(metrics.mean_squared_error(ytest, ypred))),3))


# In[21]:


Results = pd.DataFrame({'yield_actual':ytest, 'yield_pred':ypred})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = df.merge(Results, left_index=True, right_index=True)
ResultsFinal.sample(10)


# In[22]:



px.scatter(Results,'yield_pred','yield_actual',trendline='ols',trendline_color_override='blue',template='plotly_dark',title='Predicted Vs Actual Sales')


# In[23]:


import pickle 
pickle_out = open("lr.pkl","wb")
pickle.dump(lr,pickle_out)
pickle_out.close()


# In[ ]:




