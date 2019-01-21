#!/usr/bin/env python
# coding: utf-8

# ### Header Code

# In[2]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
import plotly.plotly as py
import plotly.graph_objs as go
from scipy.optimize import fmin
import math


# ## Setting Up the Bond Data

# In[3]:


curr_date = '1993-12-31'
curr_date = datetime.strptime(curr_date, '%Y-%m-%d')


# In[4]:


data = pd.read_csv('Rates_Table.txt', sep=" ", header=None)
df = data.T
df.columns = df.iloc[0]
df = df.reindex(df.index.drop(0))


# In[5]:


def calc_months(start_date, end_date):
    months = relativedelta(end_date, start_date).years*12 + relativedelta(end_date, start_date).months
    return months


# In[6]:


df['maturity'] = pd.to_datetime(df['maturity'], format='%Y%m%d')


# In[7]:


termlst = []
for i in range(0, len(df.index)):
    termlst.append(calc_months(curr_date, df['maturity'].iloc[i]))

df['monthly_term'] = termlst
df['num_payments'] = df['monthly_term'] / 6


# In[8]:


df


# ## Bootstrapping

# ### Matrix Inversion Code

# In[9]:


cf_matrix = np.zeros((10, 10))
for i in list(range(0, 10)):
    #print('i: ', i)
    for j in list(range(0, 10)):
        #print(j)
        if i >= df['num_payments'].iloc[j]:
            continue
        cf_matrix[i][j] = df['coupon'].iloc[j]/2
        #print(df['num_payments'].iloc[j])
        if i == j:
            cf_matrix[i][j] += 100
#print(cf_matrix)
bid_discount_factors = np.flip(np.linalg.inv(cf_matrix).dot(np.array(df.bid_price)))
ask_discount_factors = np.flip(np.linalg.inv(cf_matrix).dot(np.array(df.ask_price)))
pd.DataFrame([bid_discount_factors, ask_discount_factors], columns=['6 months', '12 months', '18  months', '24 months',
                                                                   '30 months', '36 months', '42 months', '48  months', 
                                                                   '54 months', '60 months'], index=['Bid DF', 'Ask DF'])


# In[10]:


df_cfmatrix = pd.DataFrame(cf_matrix)
df_cfmatrix


# ### DF Calculations: custom function

# In[96]:


def calc_discount_factors(c, p, numPay):
    discountArray = np.array([])
    for i in range(0, len(p)):
        if discountArray.size == 0:
            discountArray = np.append(discountArray, p[i]/100)
        else:
            sumDf = np.sum(discountArray)
            #print(sumDf)
            Df = (p[i] - c[i]/2*sumDf)/(100+c[i]/2)
            #print(Df)
            discountArray = np.append(discountArray, Df)
    return discountArray


askDiscountFactors = calc_discount_factors(list(df['coupon']), list(df['ask_price']), list(df['num_payments']))
bidDiscountFactors = calc_discount_factors(list(df['coupon']), list(df['bid_price']), list(df['num_payments']))
pd.DataFrame([bidDiscountFactors, askDiscountFactors], columns=['6 months', '12 months', '18  months', '24 months',
                                                                   '30 months', '36 months', '42 months', '48  months', 
                                                                   '54 months', '60 months'], index=['Bid DF', 'Ask DF'])


# #### Answers:

# In[97]:


#Question 1
#(b): Ask DF for 5 year year bond
print('(a): DF for (0,5): ', askDiscountFactors[-1])
#(a): 2 year implied bid risk free rate
c = (df['bid_price'].iloc[3] - 100*bidDiscountFactors[3])/(50*np.sum(bidDiscountFactors[0:4]))
print('(b): 2 year implied risk free rate: ', c*100)
#(c): bid price of a 3 year US Governemnt bond paying coupon of 5% semi
cf = []
period = np.arange(.5, 3.5, 0.5)
for i in range(0, len(period)):
    if i == len(period) -1:
        cf.append(105/(1+bidDiscountFactors[i])**period[i])
    cf.append(5/(1+bidDiscountFactors[i])**period[i])
print(sum(cf))
#p = df['coupon'].iloc[5]/2*np.sum(bidDiscountFactors[0:6]) + (100+df['coupon'].iloc[5]/2)*bidDiscountFactors[6]
print('(c): 3Y, 5% semiannual rate bid price: ', p)


# #### Bootstrapping: Graph of correct Discount factors

# In[13]:


trace0 = go.Scatter(
    x = list(df.monthly_term),
    y = bidDiscountFactors,
    #y = bid_discount_factors,
    mode = 'lines',
    name = 'Bid-DF'
)
trace1 = go.Scatter(
    x = list(df.monthly_term),
    y = askDiscountFactors,
    #y = ask_discount_factors,
    mode = 'lines',
    name = 'Ask-DF'
)
data = [trace0, trace1]
layout = dict(title = 'Discount Curves',
              xaxis = dict(title = 'Month'),
              yaxis = dict(title = 'Discount Rate'),
              )
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='bid-ask-DF_Curve')


# ## Nelson Siegel Method

# In[20]:


from scipy.optimize import minimize


# In[111]:


def NSmin(parameters):
    price = list(df['bid_price'])
    T = np.arange(0.5, 5.5, 0.5)#builds the time factor array [0.5, 1.0, 1.5....] by half years
    coupons = list(df['coupon'])#list of coupons from data
    discounts = np.array([])#empty array of discounts that is constantly added to
    NSerrors = np.array([])#empty array of errors
    
    for i in range(0, len(coupons)):
        #rate function that takes in parameters and Time values
        r = parameters[0] + (parameters[1] + parameters[2]) * ((1-np.exp(-T[i]/parameters[3]))/(T[i]/parameters[3])) - parameters[2] * np.exp(-T[i]/parameters[3])
        #discount factor formula that takes in Rate function and Time values
        discount_factor = np.exp(-r*T[i])
        discounts = np.append(discounts, discount_factor)
        print(discounts)
        #handles the first zero coupon bond case
        if numPayments[i] == 1:
            pmodel = 100*(discount_factor)
            NSerrors = np.append(NSerrors, (pmodel - price[i])**2)
        #handles the rest of the coupon bond cases
        else:
            pmodel = 100*(coupons[i]/2*(np.sum(discounts)) + discounts[-1])
            print(pmodel)
            NSerrors = np.append(NSerrors, (pmodel - price[i])**2)
        #NS = np.append(NS, (pmodel - price[i])**2)
    #print(NS)
    #print(discounts)
    return np.sum(NSerrors)


# In[112]:


parameters = np.array([0.1, 0.1, 0.1, 1.0])
NSmin(parameters=parameters)


# In[106]:


res = minimize(fun=NSmin, x0=parameters)

