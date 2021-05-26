#!/usr/bin/env python
# coding: utf-8

# # Business Problem

# To ensure the safety and reliability of each and every unique car configuration before they hit the road, Daimler’s engineers have developed a robust testing system. But, optimizing the speed of their testing system for so many possible feature combinations is complex and time-consuming. Hence, Daimler has challenged to reduce the time that cars spend on the test bench.
# The Objective of the Case Study is to optimize the testing process in a greener way i.e. reducing the testing time with lower carbon dioxide emissions without reducing Daimler’s standards on safety and efficiency.

# # ML Formulation

# We can pose this problem as a regression problem to predict the testing time by selecting some important features from the dataset to tackle the curse of dimensionality.
# 
# In order to know how our ML model is performing better, We will develop a baseline or random model and we will compare our models with the baseline model, to get a knowledge where our model stands.
# 

# # Performance Metric

# The metric we will use to evaluate our models is - R^2

# ### Why R^2 as metric?

# #### What is R^2
# 
# It is the amount of the variation in the output dependent attribute which is predictable from the input independent variable. It is used to check how well-observed results are reproduced by the model, depending on the ratio of total deviation of results described by the model.
# 
# #### Interpretation - 
# Assume R2 = 0.68
# It can be referred that 68% of the changeability of the dependent output attribute can be explained by the model while the remaining 32 % of the variability is still unaccounted for.
# 
# R-squared = 1 - ( SSres / SStot )
# 
# SSres is the sum of squares of the residual errors.
# 
# SStot is the total sum of the errors.
# 
# which means we scale our simple MSE based on the difference of actual values from their mean.
# 
# R^2 is a convenient rescaling of MSE that is unit invariant.
# 
# It is also very interpretable as -
# 
# The best possible score is 1 which is obtained when the predicted values are the same as the actual values.
# 
# R^2 with value 0 means the model is same as simple mean model.
# 
# Negative value of R^2 mean that the model is worse than simple mean model.
# 
# 
# #### Why?
# 
# MSE or MAE penalizes the large prediction errors hence the sum of errors can become very large and interpreting it won't be trivial.
# 
# Whereas,
# R^2 is a scale-free score i.e. irrespective of the values being small or large, the value of R square will be less than one or in worst cases just greater than 1.
# 

# # EDA

# In[2]:


'''importing dependencies'''
import pandas as pd
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from xgboost import XGBRegressor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import plotly.offline as offline
import plotly.graph_objs as go
offline.init_notebook_mode()
from collections import Counter
warnings.filterwarnings('ignore')
sns.set()


# ### Reading the data

# In[3]:


data = pd.read_csv('downloads/trainwa.csv')
data.head()


# ### Knowing the data

# #### Data Shape

# In[4]:


print("Train Data Shape : = ", data.shape)


# #### Detail View of Train Data

# In[5]:


data.describe()


# #### Checking for null values in the Data

# In[6]:


print("Number of missing values in Train data: ",data.isnull().sum().sum())


# #### Checking for duplicate values in the Data

# In[7]:


print(len(data[data.duplicated()]))


# #### Analyzing the Prediction Column 'y'

# In[8]:


y = data['y']
y.describe()


# In[9]:


ax = sns.boxplot(data['y'])


# ### Observations

# 1. The dataset does not have null values.
# 2. The dataset does not have duplicate values.
# 3. The Prediction column y has some outlier points

# #### We will first remove these outlier points then we will know the features

# #### Knowing the percentiles to decide the threshold to remove the outliers

# In[10]:


print("Listing all percentiles for training time: ")
print("100: ", np.percentile(data.y,100))
print("99.9: ", np.percentile(data.y,99.9))
print("99.8: ", np.percentile(data.y,99.8))
print("99.7: ", np.percentile(data.y,99.7))
print("99.6: ", np.percentile(data.y,99.6))
print("99.5: ", np.percentile(data.y,99.5))
print("99 : ", np.percentile(data.y,99))


# ### Observations -

# We can see from 99.7 percentile onwards the y value is increasing drastically.
# We decide the threshold as 99.7

# #### Removing Noise

# In[11]:


threshold = np.percentile(data.y,99.7)
outlliers = data[data['y']>=threshold]
data.drop(data[data['y']>=threshold].index, inplace = True)


# #### Data Shape after removing noise

# In[12]:


data.shape


# In[13]:


outlliers.head()


# #### Box-Plot of y after removing noise

# In[14]:


ax = sns.boxplot(data['y'])


# ### Observations

# 1. The dataset looks more cleaner now.

# #### Log Transformation Distribution of target

# In[15]:


sns.distplot(np.log(data['y']))


# #### Log Transformation Distribution of target variable in outlier data

# In[16]:


sns.distplot(np.log(outlliers['y']))


# ### Observations

# 1. Most of the y values lie between 80 and 140.
# 2. Only a bunch of values are greater than 150.
# 3. Few values are less than 80.

# ### Selecting important features for EDA

# #### Building a simple XGBoost Model to get Feature Importances

# In[17]:


y = data['y']
x = data.drop(columns = ['ID','y'], axis = 1)
#x = x.drop('id')
x.shape, y.shape
cols = x.columns


# In[18]:


x_cat = data.loc[:,'X0':'X8']
x_num = data.loc[:,'X10':]


# In[18]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
for i in x_cat.columns:
    x_cat[i] = enc.fit_transform(x_cat[i])


# In[19]:


x = pd.DataFrame(np.hstack((x_cat,x_num)), columns = cols)


# In[20]:


x.head()


# In[21]:


model = XGBRegressor(n_estimators=100, learning_rate = 0.1,n_jobs = -1)
model.fit(x,y)


# In[22]:


imp = pd.DataFrame()
imp['columns'] = x.columns
imp['importances'] = model.feature_importances_[0]
result = imp.sort_values(by = 'importances')[:10]


# #### Top 10 important features with importances

# In[23]:


result


# In[24]:


print("The Top 10 important features are : ", list(result['columns']))


# ### EDA for important Features

# #### Knowing important features

# In[25]:


imp_features = list(result['columns'])
imp_data = pd.DataFrame()
for i in imp_features:
    imp_data[i] = x[i]


# In[26]:


#plotting categorical columns 
fig, ax = plt.subplots(5, 2, figsize=(20, 20))
for variable, subplot in zip(imp_data.columns, ax.flatten()):
    sns.countplot(imp_data[variable], ax=subplot)


# ### Observations - 

# 1. The Categorical Feature X0 has well distributed Categories.
# 
# 2. For the binary features X261 and X263 have significant values of 1 while the rest have mostly 0 values, which means tey are sparse.

# #### Univariate Analysis for Important features

# #### Box - Plot

# In[27]:


fig, ax = plt.subplots(5, 2, figsize=(60, 60))
for variable, subplot in zip(imp_data.columns, ax.flatten()):
    sns.boxplot(x = data[variable],y = data['y'], ax=subplot)


# ### Observations - 

# 1. For binary features X263, X261, X255, X260, X259 the testing time is distinguishable according to the 0 and 1 value.
# 
# 
# 2. For X0 the percentiles for all the categories is distinguishable so we can interpret the testing time acoording to category.
# 
# 
# 3. For X263, 0 value indicates testing time less than 85 and 1 value indicates testing time between 80 and 130.
# 
# 
# 4. For X261, 0 value indicates testing time between 75 and 110 and 1 value indicates testing time between 100 and 130.
# 

# #### Line Plot

# In[28]:


fig, ax = plt.subplots(5, 2, figsize=(60, 60))
for variable, subplot in zip(imp_data.columns, ax.flatten()):
    sns.lineplot(x = data[variable],y = data['y'], ax=subplot)


# ### Observations - 

# 1. Line Plot is very interpretable for X0 which has many categories, while for binary features this does not give better information than Box-Plot.
# 
# 
# 2. For X0 we can conclude that, category 'aa' results in testing time greater than 130 while categories 'az' and 'bc' result in less than 80. The rest lie between 80 and 130.

# In[ ]:





# In[ ]:





# #### Bi - Variate Analysis of Important Features

# ### Co - relation of features

# ### Chi Squared Test

# In[29]:


import scipy.stats as stats
rows = imp_data.columns
col =  imp_data.columns
chi2_matrix = pd.DataFrame(columns = col, index = rows)
p_matrix = pd.DataFrame(columns = col, index = rows)
lesser_correlated_cols = []
for i in imp_features:
    for j in imp_features:
        if i != j:
            table = pd.crosstab(imp_data[i],imp_data[j])
            #Observed value 
            obs_val = table.values
            chi2,p,dof,exp = stats.chi2_contingency(table)
            chi2_matrix[i][j] = chi2
            p_matrix[i][j] = p
            if p>=0.05:
                if (j,i) not in lesser_correlated_cols:
                    lesser_correlated_cols.append((i,j))

chi2_matrix = chi2_matrix.fillna(0)
print("The less realated column pairs are : ")
print(lesser_correlated_cols)
print("The heatmap for chi square values : ")
print(sns.heatmap(chi2_matrix))


# In[30]:


p_matrix = p_matrix.fillna(1)
print("The P-value matrix is :")
print(sns.heatmap(p_matrix))


# ### Observations - 

# 1. The pair of less related features are : 
# 
#     ('X0', 'X260'), ('X0', 'X259'), ('X0', 'X257'), ('X263', 'X262'), ('X263', 'X258'), ('X263', 'X255'), ('X262', 'X261'), ('X262', 'X256'), ('X262', 'X255'), ('X261', 'X260'), ('X261', 'X259'), ('X261', 'X257'), ('X261', 'X255'), ('X260', 'X256'), ('X259', 'X256'), ('X258', 'X256'), ('X257', 'X256')
# 
# 
# 2. The above pairs are decided on the basis of p-value, the null hypothesis that the features are not related is accepted.

# ### Now lets see how these lesser corelated column pairs impact the target together

# #### Feature Pairs with XO

# In[31]:


'''function to calculate mean of testing time for each pair of categories in the less co related features '''
def mean_with_category(i):
    col1 = i[0]
    col2 = i[1]
    means = {}
    for j in data[col1].unique():
        for k in data[col2].unique():
            temp = data[data[col1] == j]
            temp = temp[temp[col2] == k]
            if not temp.empty:
                target_mean = temp['y'].mean()
                means[(j,k)] = target_mean
    return means
            


# In[32]:


for i in lesser_correlated_cols[:3]:
    ans = mean_with_category(i)
    x_plot = ans.keys()
    y_plot = ans.values()
    
    plt.figure(figsize=(10, 6))
    plt.scatter([str(a) for a in x_plot], y_plot)
    plt.xticks(rotation='vertical')
    plt.xlabel("Combination of "+i[0]+ " and "+i[1])
    plt.ylabel("Target mean for each combination")
   


# ### Observations - 

# 1. For the combination X0 with X260, X259, X257, the category 'aa' and value 0 has highest testing time mean and 'bc' and 0 has lowest. 
# 

# ### Binary-Binary features

# In[33]:


fig, ax = plt.subplots(14, 1, figsize=(5, 40))
for i ,subplot in zip(lesser_correlated_cols[3:], ax.flatten()):
    ans = mean_with_category(i)
    x_plot = ans.keys()
    y_plot = ans.values()
    subplot.scatter([str(a) for a in x_plot], y_plot)
    subplot.set_title("Combination of "+i[0]+ " and "+i[1]+" VS testing Time")
    fig.tight_layout(pad=3.0)


# ### Observations - 

# 1. For X263 and X262,
# 
#     if both have 0 values the average testing time is less than 80.
#     if X263 has 1 and X262 has 0, the average testing time is more than 100.
#     if both have value 1, the average testing time is more than 105.
# 
# 
# 2. For X263 and X258,
# 
#     if both have 0 values the average testing time is less than 80.
#     if X263 has 1 and X258 has 0, the average testing time is close to 100.
#     if both have value 1, the average testing time is more than 100.
# 
# 
# 3. For X263 and X255,
# 
#     if both have 0 values the average testing time is less than 80.
#     if X263 has 1 and X255 has 0, the average testing time is close to 100.
#     if both have value 1, the average testing time is more than 110.
# 
# 
# 4. For X262 and X261,
# 
#     if both have 0 values the average testing time is less than 95.
#     if X262 has 1 and X261 has 0, the average testing time is close to 105.
#     if X262 has 0 and X261 has 1, the average testing time is more than 105.
# 
# 
# 5. For X262 and X256,
# 
#     if both have 0 values the average testing time is close to 101.
#     if X262 has 1 and X256 has 0, the average testing time is close to 104.
#     if X262 has 0 and X261 has 1, the average testing time is more than 94.
# 
# 
# 6. For X262 and X255,
# 
#     if both have 0 values the average testing time is close to 100.
#     if X262 has 1 and X255 has 0, the average testing time is close to 104.
#     if X262 has 0 and X255 has 1, the average testing time is more than 110.
# 
# 
# 7. For X261 and X260,
# 
#     if both have 0 values the average testing time is less than 100.
#     if X261 has 1 and X260 has 0, the average testing time is close to 110.
#     if both have value 1, the average testing time is more than 120.
# 
# 
# 8. For X261 and X259,
# 
#     if both have 0 values the average testing time is close to 95.
#     if X261 has 0 and X259 has 1, the average testing time is less than 90.
#     if X261 has 1 and X259 has 0, the average testing time is close to 110.
#  
#  
# 9. For X261 and X257,
# 
#     if both have 0 values the average testing time is close to 95.
#     if X261 has 0 and X257 has 1, the average testing time is less than 105.
#     if X261 has 1 and X257 has 0, the average testing time is close to 110.
# 
# 
# 10. For X261 and X255,
# 
#     if both have 0 values the average testing time is close to 95.
#     if X261 has 0 and X255 has 1, the average testing time is more than 110.
#     if X261 has 1 and X257 has 0, the average testing time is close to 110.
#     if both have value 1, the average testing time is close to 110.
#  
#  
# 11. For X260 and X256,
# 
#     if both have 0 values the average testing time is close to 100.
#     if X260 has 0 and X256 has 1, the average testing time is less than 100.
#     if both have value 1, the average testing time is more than 120.
#  
#  
# 12. For X259 and X256,
# 
#     if both have 0 values the average testing time is close to 100.
#     if X259 has 0 and X256 has 1, the average testing time is close to 94.
#     if X259 has 1 and X256 has 0, the average testing time is more than 120.
#   
#   
# 13. For X258 and X256,
# 
#     if both have 0 values the average testing time is close to 101.
#     if X258 has 0 and X256 has 1, the average testing time is close to 94.
#     if X258 has 1 and X256 has 0, the average testing time is more than 102.
#    
#    
# 14. For X257 and X256,
# 
#     if both have 0 values the average testing time is close to 101.
#     if X257 has 0 and X256 has 1, the average testing time is close to 94.
#     if X257 has 1 and X256 has 0, the average testing time is more than 105.

# ### EDA - Conclusion

# #### 1. There are no null and duplicate values in the data.
# #### 2. Categorical features seem to hold more information as they are more widely present and also account for testing time geater                 than 130 and less than 80. Also they give information about for testing time between 80 and 130
# #### 3. The numerical features are either 0 or 1 with most of the values being 0.
# #### 4. The most important features are 'X0', 'X263', 'X262', 'X261', 'X260', 'X259', 'X258', 'X257', 'X256', 'X255'
# #### 5. The prediction column y has most values between 80 and 150.

# # Feature Engineering

# In[34]:


# split into train test sets
from sklearn.model_selection import train_test_split
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
print("Done")


# ### Baseline model 

# #### model which outputs mean

# In[35]:


y_pred_value  = y_train.mean()
y_pred = []
for i in range(0,len(y_test)):
    y_pred.append(y_pred_value)


# In[36]:


from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print(score)


# In[ ]:





# In[ ]:





# In[37]:


x.head()


# In[38]:


# split into train test sets
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
print("Done")


# In[39]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[40]:


#train autoencoder for regression with no compression in the bottleneck layer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot
from keras import backend as K


# In[41]:


n_inputs = x.shape[1]


# In[42]:


# define encoder
input_data = Input(shape=(n_inputs,))
#encoder level 1
encoder = Dense(n_inputs*2)(input_data)
encoder = BatchNormalization()(encoder)
encoder = LeakyReLU()(encoder)

# define bottleneck
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(encoder)

# decoder level 2
decoder = Dense(n_inputs*2)(bottleneck)
decoder = BatchNormalization()(decoder)
decoder = LeakyReLU()(decoder)

# output layer
output = Dense(n_inputs, activation='linear')(decoder)
# define autoencoder model
model = Model(inputs=input_data, outputs=output)
# compile autoencoder model
model.compile(optimizer='adam', loss='mse')


# In[43]:


model.summary()


# In[44]:


# plot the autoencoder
plot_model(model, 'autoencoder.png', show_shapes=True)

# fit the autoencoder model to reconstruct input
history = model.fit(X_train, y_train, epochs=400, batch_size=16, verbose=2, validation_data=(X_test,y_test))
# plot loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# define an encoder model (without the decoder)
encoder = Model(inputs=input_data, outputs=bottleneck)
plot_model(encoder, 'encoder.png', show_shapes=True)
# save the encoder to file
encoder.save('encoder.h5')


# In[45]:


from tensorflow.keras.models import load_model
# load the model from file
encoder = load_model('encoder.h5')


# In[46]:


X_train_encode = encoder.predict(X_train)
# encode the test data
X_test_encode = encoder.predict(X_test)


# In[47]:


X_train_encode.shape


# ### Random XGBoost Model with encoded Features

# In[48]:


reg = XGBRegressor(n_estimators=100, learning_rate = 0.1)
reg.fit(X_train_encode,y_train)
y_pred = reg.predict(X_test_encode)
score = r2_score(y_test, y_pred)
print(score)


# ### Observation - 

# The Random XGBoost Model with encoded features produces better r2 than Simple mean model

# ## Some Other Feature Engg Techniques

# ### PCA Features

# In[49]:


from sklearn.decomposition import PCA
#taking top 10 components
components = 10 
pca = PCA(n_components=components, random_state=420)

x_pca = pd.DataFrame(pca.fit_transform(x_num))

print(x_pca.shape)
print(x_pca.head())


# ### SVD

# In[50]:


# get the matrix factors
U, S, VT = np.linalg.svd(x_num,full_matrices=1)
# calculating the aspect ratio b
m = x_num.shape[1]
n = x_num.shape[0]
b = m/n

#taking w_b from table correspondng to b
w_b = 1.6089

# getting the median singular value
ymed = np.median(S)

# finding the  Hard threshold
cutoff = w_b * ymed 
print("The Hard Threshold for Truncation = ",cutoff)
# get the number of components
r = np.max(np.where(S > cutoff))
print("Number of total components to be selected = ",r)


# In[51]:


from sklearn.decomposition import TruncatedSVD
import pickle
n_comp = r

tsvd = TruncatedSVD(n_components=r, random_state=420)

x_svd= tsvd.fit_transform(x_num)

pickle_file_name = "SVD.pkl"  

with open(pickle_file_name, 'wb') as file:  
    pickle.dump(tsvd, file)
print(x_svd.shape)


# # Different Models

# ## XGBoost

# In[52]:


results = []


# ### Feature Set - 1

# #### Auto - Encoded Features + XGBoost

# In[53]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.33)
print("Done")


# In[54]:


X_train_encode = encoder.predict(X_train)
# encode the test data
X_test_encode = encoder.predict(X_test)
#
X_cv_encode = encoder.predict(X_cv)


# In[55]:


X_train_encode.shape


# In[56]:


from sklearn.metrics import r2_score
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] 
n_estimators = [5,10,50,75,100,200]
auc_train = []
auc_cv = []
plot_rate,plot_estim = [],[]
for i in learning_rate:
    for j in n_estimators:
        clf = XGBRegressor(learning_rate = i, n_estimators = j,verbosity = 0,n_jobs = -1) 
        clf.fit(X_train_encode ,y_train)
        y_train_pred = clf.predict(X_train_encode)
        y_cv_pred = clf.predict(X_cv_encode)
        auc_train.append(r2_score(y_train,y_train_pred))
        auc_cv.append(r2_score(y_cv,y_cv_pred))
        plot_rate.append(i)
        plot_estim.append(j)


# In[57]:


#plotting the auc corresponding to different hyper parameter permutations to understand
trace1 = go.Scatter3d(x=plot_estim,y=plot_rate,z=auc_train, name = 'train')
trace2 = go.Scatter3d(x=plot_estim,y=plot_rate,z=auc_cv, name = 'Cross validation')
data = [trace1, trace2]

layout = go.Layout(scene = dict(
        xaxis = dict(title='n_estimators'),
        yaxis = dict(title='learning_rate'),
        zaxis = dict(title='R2'),))

fig = go.Figure(data=data, layout=layout)
fig.show()


# In[58]:


model = XGBRegressor(n_estimators=50, learning_rate =0.1)
model.fit(X_train_encode,y_train)
y_te = model.predict(X_test_encode)
score1 = r2_score(y_test, y_te)
results.append(score1)
print("Test Score for 1st feature set : ", score1)


# In[ ]:





# In[ ]:





# ### Feature Set - 2

# #### Auto - Encoded Features + PCA + XGBoost

# In[59]:


X_train_Set2, X_test_Set2, y_train, y_test = train_test_split(x_pca, y, test_size=0.33, random_state=1)
X_train_Set2, X_cv_Set2, y_train, y_cv = train_test_split(X_train_Set2, y_train, test_size=0.33)
print("Done")


# In[60]:


X_train_Set2.shape


# In[61]:


X_train_Set2 = pd.DataFrame(np.hstack((X_train_encode,X_train_Set2)))
X_cv_Set2 = pd.DataFrame(np.hstack((X_cv_encode,X_cv_Set2)))
X_test_Set2 = pd.DataFrame(np.hstack((X_test_encode,X_test_Set2)))
print(X_train_Set2.shape,X_test_Set2.shape,X_cv_Set2.shape)


# In[62]:


from sklearn.metrics import r2_score
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] 
n_estimators = [5,10,50,75,100,200]
score_train = []
score_cv = []
plot_rate,plot_estim = [],[]
for i in learning_rate:
    for j in n_estimators:
        #scaling the positive weight to tackle imbalanced data
        clf = XGBRegressor(learning_rate = i, n_estimators = j,verbosity = 0,n_jobs = -1) 
        clf.fit(X_train_Set2 ,y_train)
        y_train_pred = clf.predict(X_train_Set2)
        y_cv_pred = clf.predict(X_cv_Set2)
        score_train.append(r2_score(y_train,y_train_pred))
        score_cv.append(r2_score(y_cv,y_cv_pred))
        plot_rate.append(i)
        plot_estim.append(j)


# In[63]:


#plotting the auc corresponding to different hyper parameter permutations to understand
trace1 = go.Scatter3d(x=plot_estim,y=plot_rate,z=score_train, name = 'train')
trace2 = go.Scatter3d(x=plot_estim,y=plot_rate,z=score_cv, name = 'Cross validation')
data = [trace1, trace2]

layout = go.Layout(scene = dict(
        xaxis = dict(title='n_estimators'),
        yaxis = dict(title='learning_rate'),
        zaxis = dict(title='R2'),))

fig = go.Figure(data=data, layout=layout)
fig.show()


# In[64]:


model = XGBRegressor(n_estimators=75, learning_rate =0.1)
model.fit(X_train_Set2,y_train)
y_te = model.predict(X_test_Set2)
score2 = r2_score(y_test, y_te)
results.append(score2)
print("Test Score for 2nd feature set : ", score2)


# In[ ]:





# In[ ]:





# ### Feature Set - 3

# #### PCA + SVD + XGBoost

# In[65]:


X_Set3 = pd.DataFrame(np.hstack((x_pca,x_svd)))
print(X_Set3.shape)


# In[66]:


X_train_Set3, X_test_Set3, y_train, y_test = train_test_split(X_Set3, y, test_size=0.33, random_state=1)
X_train_Set3, X_cv_Set3, y_train, y_cv = train_test_split(X_train_Set3, y_train, test_size=0.33)
print("Done")


# In[67]:


print(X_train_Set3.shape,X_test_Set3.shape,X_cv_Set3.shape)


# In[68]:


from sklearn.metrics import r2_score
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] 
n_estimators = [5,10,50,75,100,200]
score_train = []
score_cv = []
plot_rate,plot_estim = [],[]
for i in learning_rate:
    for j in n_estimators:
        #scaling the positive weight to tackle imbalanced data
        clf = XGBRegressor(learning_rate = i, n_estimators = j,verbosity = 0,n_jobs = -1) 
        clf.fit(X_train_Set3 ,y_train)
        y_train_pred = clf.predict(X_train_Set3)
        y_cv_pred = clf.predict(X_cv_Set3)
        score_train.append(r2_score(y_train,y_train_pred))
        score_cv.append(r2_score(y_cv,y_cv_pred))
        plot_rate.append(i)
        plot_estim.append(j)


# In[69]:


#plotting the auc corresponding to different hyper parameter permutations to understand
trace1 = go.Scatter3d(x=plot_estim,y=plot_rate,z=score_train, name = 'train')
trace2 = go.Scatter3d(x=plot_estim,y=plot_rate,z=score_cv, name = 'Cross validation')
data = [trace1, trace2]

layout = go.Layout(scene = dict(
        xaxis = dict(title='n_estimators'),
        yaxis = dict(title='learning_rate'),
        zaxis = dict(title='R2'),))

fig = go.Figure(data=data, layout=layout)
fig.show()


# In[70]:


model = XGBRegressor(n_estimators=50, learning_rate =0.1)
model.fit(X_train_Set3,y_train)
y_te = model.predict(X_test_Set3)
score3 = r2_score(y_test, y_te)
results.append(score3)
print("Test Score for 3rd feature set : ", score3)


# ### Feature Set - 4

# #### Label Encoded Categorical features + original Binary Features + PCA + SVD + XGBoost

# In[71]:


X_Set4 = pd.DataFrame(np.hstack((x,x_pca,x_svd)))
print(X_Set4.shape)


# In[72]:


X_train_Set4, X_test_Set4, y_train, y_test = train_test_split(X_Set4, y, test_size=0.33, random_state=1)
X_train_Set4, X_cv_Set4, y_train, y_cv = train_test_split(X_train_Set4, y_train, test_size=0.33)
print("Done")


# In[73]:


print(X_train_Set4.shape,X_test_Set4.shape,X_cv_Set4.shape)


# In[74]:


from sklearn.metrics import r2_score
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] 
n_estimators = [5,10,50,75,100,200]
score_train = []
score_cv = []
plot_rate,plot_estim = [],[]
for i in learning_rate:
    for j in n_estimators:
        clf = XGBRegressor(learning_rate = i, n_estimators = j,verbosity = 0,n_jobs = -1) 
        clf.fit(X_train_Set4 ,y_train)
        y_train_pred = clf.predict(X_train_Set4)
        y_cv_pred = clf.predict(X_cv_Set4)
        score_train.append(r2_score(y_train,y_train_pred))
        score_cv.append(r2_score(y_cv,y_cv_pred))
        plot_rate.append(i)
        plot_estim.append(j)


# In[75]:


#plotting the auc corresponding to different hyper parameter permutations to understand
trace1 = go.Scatter3d(x=plot_estim,y=plot_rate,z=score_train, name = 'train')
trace2 = go.Scatter3d(x=plot_estim,y=plot_rate,z=score_cv, name = 'Cross validation')
data = [trace1, trace2]

layout = go.Layout(scene = dict(
        xaxis = dict(title='n_estimators'),
        yaxis = dict(title='learning_rate'),
        zaxis = dict(title='R2'),))

fig = go.Figure(data=data, layout=layout)
fig.show()


# In[76]:


model = XGBRegressor(n_estimators=50, learning_rate =0.1)
model.fit(X_train_Set4,y_train)
y_te = model.predict(X_test_Set4)
score4 = r2_score(y_test, y_te)
results.append(score4)
print("Test Score for 4th feature set : ", score4)


# In[ ]:





# ### Feature Set - 5

# #### Label Encoded Categorical features + original Binary Features + SVD + XGBoost

# In[77]:


X_Set5 = pd.DataFrame(np.hstack((x,x_svd)))
print(X_Set5.shape)


# In[78]:


X_train_Set5, X_test_Set5, y_train, y_test = train_test_split(X_Set5, y, test_size=0.33, random_state=1)
X_train_Set5, X_cv_Set5, y_train, y_cv = train_test_split(X_train_Set5, y_train, test_size=0.33)
print("Done")


# In[79]:


print(X_train_Set5.shape,X_test_Set5.shape,X_cv_Set5.shape)


# In[80]:


from sklearn.metrics import r2_score
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3] 
n_estimators = [5,10,50,75,100,200]
score_train = []
score_cv = []
plot_rate,plot_estim = [],[]
for i in learning_rate:
    for j in n_estimators:
        clf = XGBRegressor(learning_rate = i, n_estimators = j,verbosity = 0,n_jobs = -1) 
        clf.fit(X_train_Set5 ,y_train)
        y_train_pred = clf.predict(X_train_Set5)
        y_cv_pred = clf.predict(X_cv_Set5)
        score_train.append(r2_score(y_train,y_train_pred))
        score_cv.append(r2_score(y_cv,y_cv_pred))
        plot_rate.append(i)
        plot_estim.append(j)


# In[81]:


#plotting the auc corresponding to different hyper parameter permutations to understand
trace1 = go.Scatter3d(x=plot_estim,y=plot_rate,z=score_train, name = 'train')
trace2 = go.Scatter3d(x=plot_estim,y=plot_rate,z=score_cv, name = 'Cross validation')
data = [trace1, trace2]

layout = go.Layout(scene = dict(
        xaxis = dict(title='n_estimators'),
        yaxis = dict(title='learning_rate'),
        zaxis = dict(title='R2'),))

fig = go.Figure(data=data, layout=layout)
fig.show()


# In[82]:


model = XGBRegressor(n_estimators=50, learning_rate =0.1)
model.fit(X_train_Set5,y_train)
y_te = model.predict(X_test_Set5)
score5 = r2_score(y_test, y_te)
results.append(score5)
print("Test Score for 5th feature set : ", score5)


# In[ ]:





# In[ ]:


col = x_num.columns
v = []
for i in x_num.columns:
    v.append(x_num[i].var())
v = np.array(v)
sns.scatterplot(col,v, hue = v)


# In[ ]:





# In[ ]:





# ## Linear Regression

# #### With Feature Set - 4

# In[83]:


from sklearn.linear_model import LinearRegression
X_Set4 = pd.DataFrame(np.hstack((x,x_pca,x_svd)))
print(X_Set4.shape)


# In[84]:


X_train_LR_Set4, X_test_LR_Set4, y_train, y_test = train_test_split(X_Set4, y, test_size=0.33, random_state=1)
print("Done")


# In[85]:


lr = LinearRegression()
lr.fit(X_train_LR_Set4,y_train)
y_pred = lr.predict(X_test_LR_Set4)
score6 = r2_score(y_test,y_pred)
print('R_2 Error on test : ', score6)


# In[ ]:





# #### With Feature Set - 5

# In[86]:


X_Set5 = pd.DataFrame(np.hstack((x,x_svd)))
print(X_Set5.shape)


# In[87]:


X_train_LR_Set5, X_test_LR_Set5, y_train, y_test = train_test_split(X_Set5, y, test_size=0.33, random_state=1)
print("Done")


# In[88]:


lr = LinearRegression()
lr.fit(X_train_LR_Set5,y_train)
y_pred = lr.predict(X_test_LR_Set5)
score7 = r2_score(y_test,y_pred)
print('R_2 Error on test : ', score7)


# In[ ]:





# ## Random Forest

# #### With Feature Set - 4

# In[89]:


X_Set4 = pd.DataFrame(np.hstack((x,x_pca,x_svd)))
print(X_Set4.shape)


# In[90]:


X_train_RF_Set4, X_test_RF_Set4, y_train, y_test = train_test_split(X_Set4, y, test_size=0.33, random_state=1)
X_train_RF_Set4, X_cv_RF_Set4, y_train, y_cv = train_test_split(X_train_RF_Set4, y_train, test_size=0.33)
print("Done")


# In[91]:


from sklearn.ensemble import RandomForestRegressor
max_depth = [5, 10, 15,20, 25, 40] 
n_estimators = [5,10,50,75,100,200]
score_train = []
score_cv = []
plot_dep,plot_estim = [],[]
for i in max_depth:
    for j in n_estimators:
        clf = RandomForestRegressor(max_depth = i, n_estimators = j, verbose = 0,n_jobs = -1) 
        clf.fit(X_train_RF_Set4 ,y_train)
        y_train_pred = clf.predict(X_train_RF_Set4)
        y_cv_pred = clf.predict(X_cv_RF_Set4)
        score_train.append(r2_score(y_train,y_train_pred))
        score_cv.append(r2_score(y_cv,y_cv_pred))
        plot_dep.append(i)
        plot_estim.append(j)


# In[92]:


#plotting the auc corresponding to different hyper parameter permutations to understand
trace1 = go.Scatter3d(x=plot_estim,y=plot_dep,z=score_train, name = 'train')
trace2 = go.Scatter3d(x=plot_estim,y=plot_dep,z=score_cv, name = 'Cross validation')
data = [trace1, trace2]

layout = go.Layout(scene = dict(
        xaxis = dict(title='n_estimators'),
        yaxis = dict(title='max_depth'),
        zaxis = dict(title='R2'),))

fig = go.Figure(data=data, layout=layout)
fig.show()


# In[93]:


model = RandomForestRegressor(n_estimators=200, max_depth =5)
model.fit(X_train_RF_Set4,y_train)
y_te = model.predict(X_test_RF_Set4)
score8 = r2_score(y_test, y_te)
print("Test Score for 4th feature set : ", score8)


# #### With Feature Set - 5

# In[94]:


X_Set5 = pd.DataFrame(np.hstack((x,x_svd)))
print(X_Set5.shape)


# In[95]:


X_train_RF_Set5, X_test_RF_Set5, y_train, y_test = train_test_split(X_Set5, y, test_size=0.33, random_state=1)
X_train_RF_Set5, X_cv_RF_Set5, y_train, y_cv = train_test_split(X_train_RF_Set5, y_train, test_size=0.33)
print("Done")


# In[96]:


max_depth = [5, 10, 15,20, 25, 40] 
n_estimators = [5,10,50,75,100,200]
score_train = []
score_cv = []
plot_dep,plot_estim = [],[]
for i in max_depth:
    for j in n_estimators:
        clf = RandomForestRegressor(max_depth = i, n_estimators = j, verbose = 0,n_jobs = -1) 
        clf.fit(X_train_RF_Set5 ,y_train)
        y_train_pred = clf.predict(X_train_RF_Set5)
        y_cv_pred = clf.predict(X_cv_RF_Set5)
        score_train.append(r2_score(y_train,y_train_pred))
        score_cv.append(r2_score(y_cv,y_cv_pred))
        plot_dep.append(i)
        plot_estim.append(j)


# In[97]:


#plotting the auc corresponding to different hyper parameter permutations to understand
trace1 = go.Scatter3d(x=plot_estim,y=plot_dep,z=score_train, name = 'train')
trace2 = go.Scatter3d(x=plot_estim,y=plot_dep,z=score_cv, name = 'Cross validation')
data = [trace1, trace2]

layout = go.Layout(scene = dict(
        xaxis = dict(title='n_estimators'),
        yaxis = dict(title='max_depth'),
        zaxis = dict(title='R2'),))

fig = go.Figure(data=data, layout=layout)
fig.show()


# In[98]:


model = RandomForestRegressor(n_estimators=200, max_depth =5)
model.fit(X_train_RF_Set5,y_train)
y_te = model.predict(X_test_RF_Set5)
score9 = r2_score(y_test, y_te)
print("Test Score for 4th feature set : ", score9)


# ## MLP

# #### With Feature Set - 4

# In[99]:


from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import datetime


# In[100]:


from keras import backend as K
"""Custom R2 Score"""
def rsquared(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


# In[101]:


X_Set4 = pd.DataFrame(np.hstack((x,x_pca,x_svd)))
print(X_Set4.shape)


# In[102]:


X_train_MLP_Set4, X_test_MLP_Set4, y_train, y_test = train_test_split(X_Set4, y, test_size=0.33, random_state=1)
print("Done")


# In[103]:


input_dim = X_train_MLP_Set4.shape[1]

# The Input Layer :
model = Sequential()
model.add(Dense(128,kernel_initializer='normal', input_dim=input_dim, activation='relu'))

# The Hidden Layers :
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# The Output Layer :
model.add(Dense(1, kernel_initializer='normal',activation='linear'))


model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rsquared])
model.summary()


# In[104]:



filepath="/tmp/checkpoint"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_rsquared', verbose=1, save_best_only=True, mode='max')

optimizer = tf.keras.optimizers.Adam(0.01)


#time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir= "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True,write_grads=True)

callbacks_list = [checkpoint,tensorboard_callback]

model.fit(X_train_MLP_Set4,y_train,epochs=200, validation_data=(X_test_MLP_Set4,y_test), batch_size=1000, callbacks=callbacks_list)


# In[105]:


score10 = 0.5587


# #### With Feature Set - 5

# In[106]:


X_Set5 = pd.DataFrame(np.hstack((x,x_svd)))
print(X_Set5.shape)


# In[107]:


X_train_MLP_Set5, X_test_MLP_Set5, y_train, y_test = train_test_split(X_Set5, y, test_size=0.33, random_state=1)
print("Done")


# In[108]:


input_dim = X_train_MLP_Set5.shape[1]

# The Input Layer :
model = Sequential()
model.add(Dense(128,kernel_initializer='normal', input_dim=input_dim, activation='relu'))

# The Hidden Layers :
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# The Output Layer :
model.add(Dense(1, kernel_initializer='normal',activation='linear'))


model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rsquared])
model.summary()


# In[109]:



filepath="/tmp/checkpoint2"
checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_rsquared', verbose=1, save_best_only=True, mode='max')

optimizer = tf.keras.optimizers.Adam(0.01)


#time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir= "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1, write_graph=True,write_grads=True)

callbacks_list = [checkpoint,tensorboard_callback]

model.fit(X_train_MLP_Set5,y_train,epochs=200, validation_data=(X_test_MLP_Set5,y_test), batch_size=1000, callbacks=callbacks_list)


# In[110]:


score11 = 0.53969


# ## Decision Tree

# In[111]:


from sklearn.tree import DecisionTreeRegressor


# #### With Feature Set - 4

# In[112]:


X_Set4 = pd.DataFrame(np.hstack((x,x_pca,x_svd)))
print(X_Set4.shape)


# In[113]:


X_train_DT_Set4, X_test_DT_Set4, y_train, y_test = train_test_split(X_Set4, y, test_size=0.33, random_state=1)
X_train_DT_Set4, X_cv_DT_Set4, y_train, y_cv = train_test_split(X_train_DT_Set4, y_train, test_size=0.33)
print("Done")


# In[114]:


max_depth = [1,5,10,50]
samples_split = [5,10,100,500]
score_train = []
score_cv = []
plot_dep,plot_sample = [],[]
for i in max_depth:
    for j in samples_split:
        clf = DecisionTreeRegressor(max_depth = i, min_samples_split = j) 
        clf.fit(X_train_DT_Set4 ,y_train)
        y_train_pred = clf.predict(X_train_DT_Set4)
        y_cv_pred = clf.predict(X_cv_DT_Set4)
        score_train.append(r2_score(y_train,y_train_pred))
        score_cv.append(r2_score(y_cv,y_cv_pred))
        plot_dep.append(i)
        plot_sample.append(j)


# In[115]:


#plotting the auc corresponding to different hyper parameter permutations to understand
trace1 = go.Scatter3d(x=plot_sample,y=plot_dep,z=score_train, name = 'train')
trace2 = go.Scatter3d(x=plot_sample,y=plot_dep,z=score_cv, name = 'Cross validation')
data = [trace1, trace2]

layout = go.Layout(scene = dict(
        xaxis = dict(title='min_samples_split'),
        yaxis = dict(title='max_depth'),
        zaxis = dict(title='R2'),))

fig = go.Figure(data=data, layout=layout)
fig.show()


# In[116]:


model = DecisionTreeRegressor(max_depth =5, min_samples_split =500 )
model.fit(X_train_DT_Set4,y_train)
y_te = model.predict(X_test_DT_Set4)
score12 = r2_score(y_test, y_te)
print("Test Score for 4th feature set : ", score12)


# #### With Feature Set - 5

# In[117]:


X_Set5 = pd.DataFrame(np.hstack((x,x_svd)))
print(X_Set5.shape)


# In[118]:


X_train_DT_Set5, X_test_DT_Set5, y_train, y_test = train_test_split(X_Set5, y, test_size=0.33, random_state=1)
X_train_DT_Set5, X_cv_DT_Set5, y_train, y_cv = train_test_split(X_train_DT_Set5, y_train, test_size=0.33)
print("Done")


# In[119]:


max_depth = [1,5,10,50]
samples_split = [5,10,100,500]
score_train = []
score_cv = []
plot_dep,plot_sample = [],[]
for i in max_depth:
    for j in samples_split:
        clf = DecisionTreeRegressor(max_depth = i, min_samples_split = j) 
        clf.fit(X_train_DT_Set5 ,y_train)
        y_train_pred = clf.predict(X_train_DT_Set5)
        y_cv_pred = clf.predict(X_cv_DT_Set5)
        score_train.append(r2_score(y_train,y_train_pred))
        score_cv.append(r2_score(y_cv,y_cv_pred))
        plot_dep.append(i)
        plot_sample.append(j)


# In[120]:


#plotting the auc corresponding to different hyper parameter permutations to understand
trace1 = go.Scatter3d(x=plot_sample,y=plot_dep,z=score_train, name = 'train')
trace2 = go.Scatter3d(x=plot_sample,y=plot_dep,z=score_cv, name = 'Cross validation')
data = [trace1, trace2]

layout = go.Layout(scene = dict(
        xaxis = dict(title='min_samples_split'),
        yaxis = dict(title='max_depth'),
        zaxis = dict(title='R2'),))

fig = go.Figure(data=data, layout=layout)
fig.show()


# In[121]:


model = DecisionTreeRegressor(max_depth =10, min_samples_split =500 )
model.fit(X_train_DT_Set5,y_train)
y_te = model.predict(X_test_DT_Set5)
score13 = r2_score(y_test, y_te)
print("Test Score for 4th feature set : ", score13)


# ## Concluding Results of different Models

# In[122]:


from tabulate import tabulate
print(tabulate([['Auto - Encoded Features','XGBoost', score1], 
                 ['Auto - Encoded Features + PCA','XGBoost', score2],
                 ['PCA + SVD','XGBoost', score3],
                 ['Label - Encoded Features + PCA + SVD','XGBoost', score4],
                 ['Label - Encoded Features + SVD','XGBoost', score5],
                ['Label - Encoded Features + PCA + SVD','Linear Regression', score6],
                ['Label - Encoded Features + SVD','Linear Regression', score7],
               ['Label - Encoded Features + PCA + SVD','Random Forest', score8],
                ['Label - Encoded Features + SVD','Random Forest', score9],
                ['Label - Encoded Features + PCA + SVD','MLP', score10],
                ['Label - Encoded Features + SVD','MLP', score11],
                ['Label - Encoded Features + PCA + SVD','Decision Tree', score12],
                ['Label - Encoded Features + SVD','Decision Tree', score13]], 
                headers=['Features', 'Model','R2_Score'], tablefmt='orgtbl'))


# # Building Final Model

# In[123]:


import pickle 


# In[124]:



enc = LabelEncoder()
for i in x_cat.columns:
    x_cat[i] = enc.fit_transform(x_cat[i])


# In[125]:


test_data = pd.read_csv('downloads/testwa.csv')


# In[126]:


test_cat = test_data.loc[:,'X0':'X8']
test_num = test_data.loc[:,'X10':]
ids = test_data['ID']


# In[127]:


for i in x_cat.columns:
    test_cat[i] = enc.fit_transform(test_cat[i])


# In[128]:


test_svd = tsvd.transform(test_num)


# In[129]:


test = pd.DataFrame(np.hstack((test_cat,test_num,test_svd)))
print(test.shape)


# In[134]:


X_final = pd.DataFrame(np.hstack((x,x_svd)))
X_final.shape


# In[135]:


X_train_final, X_cv_final, y_train, y_cv = train_test_split(X_final, y, test_size=0.33)
print("Done")


# In[136]:


max_depth = [5, 10, 15,20, 25, 40] 
n_estimators = [5,10,50,75,100,200]
score_train = []
score_cv = []
plot_dep,plot_estim = [],[]
for i in max_depth:
    for j in n_estimators:
        clf = RandomForestRegressor(max_depth = i, n_estimators = j, verbose = 0,n_jobs = -1) 
        clf.fit(X_train_final ,y_train)
        y_train_pred = clf.predict(X_train_final)
        y_cv_pred = clf.predict(X_cv_final)
        score_train.append(r2_score(y_train,y_train_pred))
        score_cv.append(r2_score(y_cv,y_cv_pred))
        plot_dep.append(i)
        plot_estim.append(j)


# In[133]:


#plotting the auc corresponding to different hyper parameter permutations to understand
trace1 = go.Scatter3d(x=plot_estim,y=plot_dep,z=score_train, name = 'train')
trace2 = go.Scatter3d(x=plot_estim,y=plot_dep,z=score_cv, name = 'Cross validation')
data = [trace1, trace2]

layout = go.Layout(scene = dict(
        xaxis = dict(title='n_estimators'),
        yaxis = dict(title='learning_rate'),
        zaxis = dict(title='R2'),))

fig = go.Figure(data=data, layout=layout)
fig.show()


# In[137]:


model = RandomForestRegressor(n_estimators=200, max_depth =5)
model.fit(X_train_final,y_train)
y_pred = model.predict(test)
#pickle_file_name = "Final_Model.pkl"  

#with open(pickle_file_name, 'wb') as file:  
    #pickle.dump(model, file)


# In[ ]:


y_pred = pd.DataFrame(list(y_pred))
y_pred.head(10)


# In[153]:


#concatenating the Dataframes to form the required format
#print(count)
ids = [str(x) for x in test_data['ID']]
#print(ids)
ids = pd.DataFrame(ids)
final_sub = pd.DataFrame(np.hstack((ids,y_pred)), columns = ['ID','y'])

#checking 
final_sub.head(10)


# In[155]:


compression_opts = dict(method='zip',archive_name='submission_Daimler.csv')  

final_sub.to_csv('submission_Daimler.zip', index=False,compression=compression_opts)  


# In[141]:





# # Final Function

# In[26]:


x_cat.columns


# In[27]:


import os
os.makedirs("//Users/abhimanyuachyut/Downloads/Label")


# In[28]:


le_set = ['X0','X1','X2','X3','X4','X5','X6','X8']


# In[30]:


from sklearn.preprocessing import LabelEncoder
for i in tqdm(le_set):
    le = LabelEncoder()
    x_cat[i] = le.fit_transform(x_cat[i])
    x_cat[i] = x_cat[i].astype('int64')
    pickle.dump(le,open("//Users/abhimanyuachyut/Downloads/label/"+i+'.pkl','wb'))
    del(le)


# In[31]:


x_cat.head()


# In[32]:


test_data = pd.read_csv('downloads/testwa.csv')
test_cat = test_data.loc[:,'X0':'X8']
test_num = test_data.loc[:,'X10':]
ids = test_data['ID']


# In[33]:


test_cat.head()


# In[34]:


import pickle


for i in tqdm(le_set):
    with open("//Users/abhimanyuachyut/Downloads/label/"+i+'.pkl', 'rb') as file:  
        leg = pickle.load(file)
        test_cat[i] = test_cat[i].map(lambda s: '<unknown>' if s not in leg.classes_ else s)
        leg.classes_ = np.append(leg.classes_, '<unknown>')
        test_cat[i] = leg.transform(test_cat[i])
        del(leg)


# In[35]:


test_cat.head()


# In[ ]:





# In[ ]:





# In[ ]:


enc = LabelEncoder()
dic = {}
for i in x_cat.columns:
    x_cat[i] = enc.fit_transform(x[i])
    pickle_file_name = "Encoder_"+str(i)+".pkl"  

    with open(pickle_file_name, 'wb') as file:  
        pickle.dump(enc, file)


# In[ ]:


test_data = pd.read_csv('downloads/testwa.csv')
test_cat = test_data.loc[:,'X0':'X8']


# In[ ]:


for j in range(len(test_data)):
    for i in x_cat.columns:
        test_data[:1]
    break
print(test_data[:1])


# In[70]:


import joblib
import pickle
def final(datapoint):
        datapoint.remove(datapoint[0])
        num_data = np.array(datapoint[8:]).reshape(1,-1)
        test_cat = np.array(datapoint[:8]).reshape(1,-1)
        le_set = ['X0','X1','X2','X3','X4','X5','X6','X8']
        test_cat = pd.DataFrame(test_cat, columns = le_set)
        pk_model = joblib.load("SVD.pkl")
        svd_data = pk_model.transform(num_data)
        for i in tqdm(le_set):
            with open("//Users/abhimanyuachyut/Downloads/label/"+i+'.pkl', 'rb') as file:  
                leg = pickle.load(file)
                test_cat[i] = test_cat[i].map(lambda s: '<unknown>' if s not in leg.classes_ else s)
                leg.classes_ = np.append(leg.classes_, '<unknown>')
                test_cat[i] = leg.transform(test_cat[i])
                del(leg)
        
        final_data = pd.DataFrame(np.hstack((test_cat,num_data,svd_data)))
        final_model = joblib.load("Final_Model.pkl")
        y_pred = final_model.predict(final_data)
        return y_pred
    


# In[72]:


for i in range(1,10):
    datapoint = list(test_data.loc[i])
    ans = final(datapoint)
    print(ans)


# In[187]:


le = LabelEncoder()
for c in x_cat.columns:
    x_cat[c] = le.fit_transform(x[c])
    print(type(test_cat[c]))
    test_cat[c] = test_cat[c].map(lambda s: '<unknown>' if s not in le.classes_ else s)
    le.classes_ = np.append(le.classes_, '<unknown>')
    test_cat[c] = le.transform(test_cat[c])


# In[188]:


x_cat.head()


# In[189]:


test_cat.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




