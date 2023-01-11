import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix

def split_train_test(df, col):
    
    seed = 42
    train, val_test = train_test_split(df, train_size=.5, random_state=seed, stratify=df[col])
    validate, test = train_test_split(val_test, train_size=.6, random_state=seed, stratify=val_test[col])
    
    return train, validate, test

def make_baseline(df, baseline, col):
    '''
    This function is used to create a column within the dataframe to make a baseline column, and then calculate the baseline accuracy. Needs to be optimized more, but functions as is currently. Make sure to use the word 'baseline' when calling function.
    '''
    
    seed = 42
    
    df[baseline] = df[col].value_counts().idxmax()    

    base = (df[col] == df[baseline]).mean()
    
    print(f'Baseline Accuracy is: {base:.3}')
    
def xy_train(train, validate, test):
    '''
    This function will separate each of my datasets (train, validate, and test) and split them further into my x and y sets for modeling. When running this, be sure to assign each of the six variables in the proper order, otherwise it will almost certainly mess up. (X_train, y_train, X_validate, y_validate, X_test, y_test).
    '''
    
    seed = 42
    
    X_train = train.drop(columns=['contract_type', 
                                  'internet_service_type', 
                                  'churn_Yes',
                                  'customer_id'])
    y_train = train.churn_Yes

    X_validate = validate.drop(columns=['contract_type', 
                                   'internet_service_type', 
                                   'churn_Yes',
                                   'customer_id'])
    y_validate = validate.churn_Yes

    X_test = test.drop(columns=['contract_type', 
                                'internet_service_type', 
                                'churn_Yes',
                                'customer_id'])
    y_test = test.churn_Yes
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test