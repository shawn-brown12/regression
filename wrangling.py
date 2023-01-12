import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from env import host, username, password
from scipy import stats
from sklearn.model_selection import train_test_split
from env import host, username, password

#----------------------------------------------------------    

def get_connection(db, user=username, host=host, password=password):
    '''
    This functions imports my credentials for the Codeup MySQL server to be used to pull data
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#----------------------------------------------------------    
    
#simply copied this for the framework, will be changed for other data
def get_zillow():
    '''
    This function will check locally if there's a zillow.csv file in the local directory, and if not, working with the 
    get_connection function, will pull the zillow dataset from the Codeup MySQL server. After that, it will also save a copy of 
    the csv locally if there wasn't one, so it doesn't have to run the query each time. Additionally, this will clean up by dataset to be ready to be split and used for modeling.
    '''
    if os.path.isfile('zillow_2017.csv'):
        
        df = pd.read_csv('zillow_2017.csv')
        df = df.drop(columns='Unnamed: 0')

        return df
    
    else:
        url = get_connection('zillow')
        query = '''
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
                       taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                JOIN propertylandusetype USING(propertylandusetypeid)
                WHERE propertylandusedesc = 'Single Family Residential';
                '''
        df = pd.read_sql(query, url)        
        df = df.dropna()

        df.bedroomcnt = df.bedroomcnt.astype(int)
        df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.astype(int)
        df.taxvaluedollarcnt = df.taxvaluedollarcnt.astype(int)
        df.yearbuilt = df.yearbuilt.astype(int)
        df.fips = df.fips.astype(int)
    
        df = df.rename(columns= {'bedroomcnt': 'bedrooms',
                             'bathroomcnt': 'bathrooms',
                             'calculatedfinishedsquarefeet': 'square_ft',
                             'taxvaluedollarcnt':'tax_value',
                             'yearbuilt':'built',
                             'taxamaount':'taxes'
                             })        
        df.to_csv('zillow_2017.csv')

        return df

#----------------------------------------------------------        
    
def train_val_test(df):
    '''
    This function takes in a DataFrame and splits it into train, validate, test subsets for our modeling phase. Does not take in a stratify option.
    '''
    seed = 42
    train, val_test = train_test_split(df, train_size=.5, random_state=seed)
    validate, test = train_test_split(val_test, train_size=.6, random_state=seed)
    
    return train, validate, test

#----------------------------------------------------------    

def compare_plots(transformed_data, train, target):
    '''
    This function will take in a train dataset and a transformed train dataset, bin them, and make a basic pairplot to visualize them side by side.
    '''
    
    plt.subplot(121)
    plt.hist(train[target], bins=25)
    plt.title('Original Data')
    plt.show()
    
    plt.subplot(122)
    plt.hist(transformed_data, bins=25)
    plt.title('Transformed Data')
    
#----------------------------------------------------------        

def chi2_report(df, col, target):
    '''
    This function is to be used to generate a crosstab for my observed data, and use that the run a chi2 test, and generate the report values from the test.
    '''
    
    observed = pd.crosstab(df[col], df[target])
    
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    alpha = .05
    seed = 42
    
    print('Observed Values\n')
    print(observed.values)
    
    print('---\nExpected Values\n')
    print(expected.astype(int))
    print('---\n')

    print(f'chi^2 = {chi2:.4f}') 
    print(f'p     = {p:.4f}')

    print('Is p-value < alpha?', p < alpha)