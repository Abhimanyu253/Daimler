#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import pickle
import numpy as np
from tqdm import tqdm
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
                
                '''For handling new labels in test data'''
                test_cat[i] = test_cat[i].map(lambda s: '<unknown>' if s not in leg.classes_ else s)
                leg.classes_ = np.append(leg.classes_, '<unknown>')
                test_cat[i] = leg.transform(test_cat[i])
                del(leg)
        
        final_data = pd.DataFrame(np.hstack((test_cat,num_data,svd_data)))
        final_model = joblib.load("Final_Model.pkl")
        y_pred = final_model.predict(final_data)
        return y_pred
   


# In[2]:


import pandas as pd
test_data = pd.read_csv('downloads/testwa.csv')
pred = []
for i in range(len(test_data)):
    datapoint = list(test_data.loc[i])
    ans = final(datapoint)
    pred.append(ans[0])


# In[4]:


print("The predicted Values are : ")
print(list(pred))


# In[ ]:




