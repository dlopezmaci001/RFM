# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 19:57:06 2020

@author: Daniel
"""

# Load libraries

#%pylab inline

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import progressbar 

import warnings
warnings.filterwarnings('ignore')

def RFM_nimerya(filename,date_column,idcustomer,sales_column):
    
    """
    This function automatically calculates RFM values for a given set 
    of variables. 
    It contains 3 params. 
    filename = name of file to analyse. If path has not been set (using os)
               include path in name. File separator must be "|",
    date_column = introduce the name of the column that contains the date values.
    idcustomer = specify the name of the column having the customer_id.
    sales_column = column with the purchase amount made by customers.
    """
    
    pbar = progressbar.ProgressBar().start()
           
    # load data
    
    df = pd.read_csv(filename,sep='|',decimal=",")
    # trasnform
    
    pbar.update((1/7)*100)

    import datetime as dt
    
    df[date_column] = pd.to_datetime(df[date_column]).dt.date
    
    df = df[pd.notnull(df[idcustomer])]
       
    print(df[date_column].min(), df[date_column].max())
    
    pbar.update((2/7)*100)
    
    #Number of days to study
    
    print('Please introduce since which year you want the RFM be calculated'
          ) 
    
    year  = int(input())
    
    print('Please introduce since which month you want the RFM be calculated')
        
    month  = int(input())
    
    print('Please introduce since which day you want the RFM be calculated.')
        
    day  = int(input())
    
    sd = dt.date(year,month,day)
    
    df['total_days']=sd - df[date_column]
    
    df['total_days'].astype('timedelta64[D]')
    
    df['total_days']=df['total_days'] / np.timedelta64(1, 'D')
    
    df.head()
    
    pbar.update((3/7)*100)
    
    # We will only consider transactions from the last year
    
    df=df[df['total_days'] < 365]
    
    df.info()
    
    """
    The data will be summarized at customer level by taking number of days to the 
    latest transaction, sum of all transction amount and total number of 
    transaction.
    
    """
    
    rfmTable = df.groupby(idcustomer).agg({'total_days': lambda x:x.min(), # Recency
                                            idcustomer: lambda x: len(x),  # Frequency
                                            sales_column: lambda x: x.sum()})      # Monetary Value
    
    rfmTable.rename(columns={'total_days': 'recency', 
                              idcustomer: 'frequency', 
                              sales_column: 'monetary_value'}, inplace=True)
    
    """
    RFM analysis involves categorising R,F and M into 3 or more categories.
      For convenience, let's create 10 categories based on quartiles 
      (quartiles roughly divide the sample into 10 segments equal proportion).
    
    """

    pbar.update((4/7)*100)
    
    quartiles = rfmTable.quantile(q=[0.25,0.50,0.75])
    print(quartiles, type(quartiles))
    
    """
    let's convert quartile information into a dictionary so that cutoffs can be
    picked up.
    
    """
    quartiles=quartiles.to_dict()
    
    quartiles
    
    pbar.update((5/7)*100)
    
    # Categorise
    
    ## for Recency 
    
    def RClass(x,p,d):
        if x <= d[p][0.25]:
            return 1
        elif x <= d[p][0.50]:
            return 2
        elif x <= d[p][0.75]: 
            return 3
        else:
            return 4
        
    ## for Frequency and Monetary value 
    
    def FMClass(x,p,d):
        if x <= d[p][0.25]:
            return 4
        elif x <= d[p][0.50]:
            return 3
        elif x <= d[p][0.75]: 
            return 2
        else:
            return 1    
    
    rfmSeg = rfmTable
    rfmSeg['R_Quartile'] = rfmSeg['recency'].apply(RClass, args=('recency',quartiles,))
    rfmSeg['F_Quartile'] = rfmSeg['frequency'].apply(FMClass, args=('frequency',quartiles,))
    rfmSeg['M_Quartile'] = rfmSeg['monetary_value'].apply(FMClass, args=('monetary_value',quartiles,))
    
    """
    Now that we have the assigned numbers, let's use K-means to determine 
    the number of clusters 
    
    """
    from sklearn.cluster import KMeans
    
    cluster = KMeans(n_clusters=4)
    
    # slice matrix so we only include the 0/1 indicator columns in the clustering
    
    rfmSeg['cluster'] = cluster.fit_predict(rfmSeg[rfmSeg.columns[2:]])
    
    rfmSeg.cluster.value_counts()
    
    """
    As we can see there is not a clear distinction for the members so we will use
    another segmentation technique
    
    """
    rfmSeg['Total Score'] = rfmSeg['R_Quartile'] + rfmSeg['F_Quartile'] +rfmSeg['M_Quartile']
    
    print(rfmSeg.head(), rfmSeg.info())
    
    rfmSeg.groupby('Total Score').agg('monetary_value').mean()
    
    # Check
    
    # Monetary value
    rfmSeg.groupby('Total Score').agg('monetary_value').mean().plot(kind='bar', colormap='Blues_r')
    
    # Frequency
    
    rfmSeg.groupby('Total Score').agg('frequency').mean().plot(kind='bar', colormap='Blues_r')
    
    # Recency
    
    rfmSeg.groupby('Total Score').agg('recency').mean().plot(kind='bar', colormap='Blues_r')
    
    pbar.update((6/7)*100) 
    
    # Rescale values from 1 - 10 
    
    rfmSeg['Total Score'] = np.interp(rfmSeg['Total Score'], 
          (rfmSeg['Total Score'].min(), 
            rfmSeg['Total Score'].max()), (1, 10))
    
    rfmSeg.reset_index(idcustomer, inplace=True)
    
    """ 
    
    Now that we have structured the different customers into groups let's see if 
    distribution is validated by the pareto distribution
    
    """
    
    rfmSeg['cumulative_sum'] = rfmSeg.monetary_value.cumsum()
    
    rfmSeg['cumulative_perc'] = 100*rfmSeg.cumulative_sum/rfmSeg.monetary_value.sum()
    
    rfmSeg.groupby(idcustomer).agg('cumulative_perc').mean().plot(kind='bar', colormap='Blues_r')
    
    pbar.update((7/7)*100)
    
    rfmSeg.to_csv('rfm_result.csv', sep='|',index=False)
    
    pass 
    

  
