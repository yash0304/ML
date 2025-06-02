import pandas as pd

def get_boston():
    '''
    This method can be called for getting boston dataset from data folder.
    it can work on github as well just make sure this csv is available in data folder.
    '''
    return pd.read_csv("data/boston.csv")
    
def get_feature_df(dataframe, target_column_tobe_dropped):
    '''
    This method can be called for getting features of any data frame by dropping target variable.\n\n
    takes variables as dataframe and target variable which will be dropped. \n\n
    returns dataframe without target variable. 
    '''
    return dataframe.drop(columns = target_column_tobe_dropped,axis=1)   

def get_target_df(df,target_column):
    '''
    This method can be called for setting target variable in a new dataframe which can be easily and agnostically used in the code.
    '''
    return df[target_column]
    
def get_startup_df():
    '''
    This method is used to call 50 startups dataset, decent data set for classification problems.
    '''
    return pd.read_csv("data/50_Startups.csv")
    
def get_churn_df():
    '''
    This method is used to call telecom churn dataset to see the behaviour of what causes people to churn.\n
    Good dataset for classification problems.
    '''
    return pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    



