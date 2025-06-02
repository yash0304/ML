#importing all the necessary libraries
import pandas as pd
import eda as eda
import dataframe as df
import visualize as v
import seaborn as sns
import matplotlib.pyplot as plt
import train as t
import os
import sys

# Set working dir to the script's folder
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)

boston_df = df.get_boston() # fetching boston dataset
boston_df.head() # fetching first few columns of boston dataset

#checking null values and information about columns
boston_df.info() # no null values

#creating x and y component
boston_df_features = df.get_feature_df(boston_df,['Price'])
boston_df_target = df.get_target_df(boston_df,['Price'])

eda.get_numerical_cols(boston_df_features)#fetching numerical columns
#get_lmplot(boston_df,boston_df_features,'Price') - #giving correct result 
#eda.get_heatmap(boston_df) # giving correct result for continuous target variable
eda.get_pairgrid(boston_df,boston_df_features,boston_df_target)#pairgrid for complete relationship between variables

#training output
num_cols, cat_cols = t.get_cols(boston_df_features)
y_test,y_pred = t.model_pipeline(df.get_feature_df(boston_df,['Price']),df.get_target_df(boston_df,['Price']),num_cols=num_cols,cat_cols=cat_cols)

y_pred=y_pred.ravel() #ravel of ndarray
y_test = y_test.values.ravel() #method of ravel on data frame

v.plot_actual_vs_predicted(y_test=y_test,y_pred=y_pred) #plot on predictions vs. actuals

v.plot_residuals(y_test,y_pred) #residuals plot - good model shows congestion around 0