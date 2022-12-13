#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn
import statsmodels.api as sm


# In[2]:


data = pd.read_csv('ex4stats.csv')


# In[3]:


def estimate_b(y,x):
    reg = sm.Logit(y,x).fit()
    b_hat = reg.params
    return b_hat


# In[4]:


def log_likelihood(b_hat,x,y):
    ll_value = 0
    for i,row in enumerate(x):
        odds = np.matmul(b_hat.T, x[i,:])
        p = np.exp(odds)
        ll_value += y[i]*odds - np.log(1+p)
    return ll_value


# In[5]:


def calc_estimator_variance(b_hat,x):
    v = np.eye(x.shape[0],x.shape[0])
    for i,row in enumerate(x):
        pi = chd_expected_value(row,b_hat)
        v[i,i] = pi * (1-pi)
    mat = np.matmul(x.T,np.matmul(v,x))
    var_b =  np.linalg.inv(mat)
    return var_b


# In[6]:


def calc_p(value):
    m = np.exp(value)
    pi = m / (1+m)
    return pi


# In[7]:


def chd_expected_value(observation, b_hat):
    value = np.matmul(b_hat.T,observation)
    pi = calc_p(value)
    return pi


# In[8]:


def chd_expected_value_CI(x_new,y_new,var_b):
    value = np.matmul(x_new.T,np.matmul(var_b,x_new)) 
    interval = stats.norm.ppf(0.975)*((value)**0.5)
    CI = [calc_p(y_new - interval) , calc_p(y_new + interval)]
    return CI


# In[9]:


def Q6():
    y = data['chd'].to_numpy()
    x_mat = data.loc[:,['BMI','alcohol','age']].to_numpy()
    x = np.c_[np.ones(x_mat.shape[0]), x_mat]
    observation = np.array([1,27,8,50])
    pd.set_option('display.max_colwidth',None)
    title = ['Logistic Regression Model Equation', 'Beta Estimator','Log Likelihood Value for Beta Estimator', 'Variance of Beta Estimator','Forecast for Expected Value of new observation','CI for Expected Value Forecast of new observation']
    Question_Part = ['Question 6.a','Question 6.b','Question 6.c','Question 6.d','Question 6.e','Question 6.f']
    
    logistic_regression = f'y = e ^ (b0 + x1 * b_age + x2 * b_obesity + x3 * b_alcohol) / ( 1 + e ^ (b0 + x1 * b_age + x2 * b_obesity + x3 * b_alcohol))'
    b_hat = estimate_b(y,x)
    log_likelihood_value = log_likelihood(b_hat,x,y)
    b_hat_variance = calc_estimator_variance(b_hat,x)
    observation_forecast = chd_expected_value(observation, b_hat)
    observation_forecast_CI = chd_expected_value_CI(observation,observation_forecast,b_hat_variance)
    results = [logistic_regression,b_hat.round(5),log_likelihood_value,b_hat_variance.round(5),observation_forecast,observation_forecast_CI]
    answers = pd.DataFrame({'Question Part':Question_Part,'Required Value':title,'Result':results})
    answers = answers.set_index(['Question Part','Required Value'])
    
    display(answers)


# In[10]:


Q6()


# In[ ]:




