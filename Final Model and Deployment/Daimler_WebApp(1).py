#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np


# In[2]:


st.title("Daimler Greener Manufacturing Challenge")


# In[3]:


@st.cache
def load_data():
    test_data = pd.read_csv('Daimler/testwa.csv')
    return test_data

data_load_state = st.text("Loading Data....")
data = load_data()
data_load_state.text("Data Loaded!!")

if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(data)


# In[ ]:


import joblib
import pickle
from tqdm import tqdm
def final(datapoint):
        datapoint.remove(datapoint[0])
        num_data = np.array(datapoint[8:]).reshape(1,-1)
        test_cat = np.array(datapoint[:8]).reshape(1,-1)
        le_set = ['X0','X1','X2','X3','X4','X5','X6','X8']
        test_cat = pd.DataFrame(test_cat, columns = le_set)
        pk_model = joblib.load("Daimler/SVD.pkl")
        svd_data = pk_model.transform(num_data)
        for i in tqdm(le_set):
            with open("label/"+i+'.pkl', 'rb') as file:  
                leg = pickle.load(file)
                test_cat[i] = test_cat[i].map(lambda s: '<unknown>' if s not in leg.classes_ else s)
                leg.classes_ = np.append(leg.classes_, '<unknown>')
                test_cat[i] = leg.transform(test_cat[i])
                del(leg)
        
        final_data = pd.DataFrame(np.hstack((test_cat,num_data,svd_data)))
        final_model = joblib.load("Daimler/Final_Model.pkl")
        y_pred = final_model.predict(final_data)
        return y_pred


# In[ ]:


if st.checkbox("Run Model for prediction"):
    st.subheader("The predicted Testing Times are:")
    for i in range(len(data)):
        datapoint = list(data.loc[i])
        ans = final(datapoint)
        st.text(ans)

