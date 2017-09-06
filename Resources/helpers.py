#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 13:52:36 2017

@author: ckadar
"""
from scipy.stats import skew
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import numpy as np
import sklearn.preprocessing as pp 

#%%
def mk_number(s):
    if type(s) is str:
        s = s.strip()
        return float(s) if s else 0
    else:
        return s
        
def mk_float(s):
    s = s.strip()
    return float(s) if s else 0

def mk_int(s):
    s = s.strip()
    return int(s) if s else 0

def smooth_float(s):
     if (type(s) is float):
        return (1+s)
     else:
        return (1+mk_float(s))

# !!! natural log
def smooth_log(a):
    if type(a) is float:
        return np.log(1+a)
    else:
        return np.log(1+mk_int(a))

        
def analyse_and_transform(x_series):
    x = np.array(x_series)
    x_std = pp.scale(x)
    x_bc , lambda_  = boxcox(1+x)
    x_bc_std = pp.scale(x_bc)
    print(lambda_)
    
    print(type(x))
    print(type(x_std))
    print(type(x_bc))
    print(x_bc)
    print(x_bc_std)

    x_n = len(x)
    x_min = np.min(x)
    x_q1 = np.percentile(x,25)
    x_median = np.median(x)
    x_mean = np.mean(x)
    x_q3 = np.percentile(x,75)
    x_max = np.max(x)
    x_stdev = np.std(x)
    x_skness = skew(x)
    
    x_std_n = len(x_std)
    x_std_min = np.min(x_std)
    x_std_q1 = np.percentile(x_std,25)
    x_std_median = np.median(x_std)
    x_std_mean = np.mean(x_std)
    x_std_q3 = np.percentile(x_std,75)
    x_std_max = np.max(x_std)
    x_std_stdev = np.std(x_std)
    x_std_skness = skew(x_std)
    
    x_bc_n = len(x_bc)
    x_bc_min = np.min(x_bc)
    x_bc_q1 = np.percentile(x_bc,25)
    x_bc_median = np.median(x_bc)
    x_bc_mean = np.mean(x_bc)
    x_bc_q3 = np.percentile(x_bc,75)
    x_bc_max = np.max(x_bc)
    x_bc_stdev = np.std(x_bc)
    x_bc_skness = skew(x_bc)
    
    x_bc_std_n = len(x_bc_std)
    x_bc_std_min = np.min(x_bc_std)
    x_bc_std_q1 = np.percentile(x_bc_std,25)
    x_bc_std_median = np.median(x_bc_std)
    x_bc_std_mean = np.mean(x_bc_std)
    x_bc_std_q3 = np.percentile(x_bc_std,75)
    x_bc_std_max = np.max(x_bc_std)
    x_bc_std_stdev = np.std(x_bc_std)
    x_bc_std_skness = skew(x_bc_std)
    
    print("X")
    print ("number of instances: " , x_n)
    print ("minimum: ", x_min)
    print ("Q1: ", x_q1)
    print ("median: ", x_median)
    print ("mean: ", x_mean)
    print ("Q3: ", x_q3)
    print ("maximum: ", x_max)
    print ('standard deviation: ', x_stdev)
    print ("skewness: ", x_skness)
    
    plt.hist(x) 
    plt.xlabel("X") 
    plt.ylabel("Frequency") 
    plt.title("X Histogram") 
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    plt.boxplot(x)
    plt.ylabel("Value") 
    plt.title("X Boxplot")
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    #####
    
    print("X Standardized")
    print ("number of instances: " , x_std_n)
    print ("minimum: ", x_std_min)
    print ("Q1: ", x_std_q1)
    print ("median: ", x_std_median)
    print ("mean: ", x_std_mean)
    print ("Q3: ", x_std_q3)
    print ("maximum: ", x_std_max)
    print ('standard deviation: ', x_std_stdev)
    print ("skewness: ", x_std_skness)
    
    plt.hist(x_std) 
    plt.xlabel("X Standardized") 
    plt.ylabel("Frequency") 
    plt.title("X Standardized Histogram") 
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    plt.boxplot(x_std)
    plt.ylabel("Value") 
    plt.title("X Standardized Boxplot")
    plt.tight_layout()
    plt.show()
    
    #####
    
    print("X Transformed")
    print ("number of instances: " , x_bc_n)
    print ("minimum: ", x_bc_min)
    print ("Q1: ", x_bc_q1)
    print ("median: ", x_bc_median)
    print ("mean: ", x_bc_mean)
    print ("Q3: ", x_bc_q3)
    print ("maximum: ", x_bc_max)
    print ('standard deviation: ', x_bc_stdev)
    print ("skewness: ", x_bc_skness)
    
    plt.hist(x_bc) 
    plt.xlabel("X Transformed") 
    plt.ylabel("Frequency") 
    plt.title("X Transformed Histogram") 
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    plt.boxplot(x_bc)
    plt.ylabel("Value") 
    plt.title("X Transformed Boxplot")
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    
    #####
    
    print("X Transformed Standardized")
    print ("number of instances: " , x_bc_std_n)
    print ("minimum: ", x_bc_std_min)
    print ("Q1: ", x_bc_std_q1)
    print ("median: ", x_bc_std_median)
    print ("mean: ", x_bc_std_mean)
    print ("Q3: ", x_bc_std_q3)
    print ("maximum: ", x_bc_std_max)
    print ('standard deviation: ', x_bc_std_stdev)
    print ("skewness: ", x_bc_std_skness)
    
    plt.hist(x_bc_std) 
    plt.xlabel("X Transformed Standardized") 
    plt.ylabel("Frequency") 
    plt.title("X Transformed Histogram") 
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    plt.boxplot(x_bc_std)
    plt.ylabel("Value") 
    plt.title("X Transformed Standardized Boxplot")
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    #print(type(x_bc_std))
    return x_bc_std
    
    
    
    