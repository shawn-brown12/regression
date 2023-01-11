import os
import pandas as pd
from env import host, username, password
from scipy import stats
from sklearn.model_selection import train_test_split
from env import host, username, password

def get_connection(db, user=username, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

#simply copied this for the framework, will be changed for other data
def get_zillow():
    '''
    This function will check locally if there's a zillow.csv file in the local directory, and if not, working with the 
    get_connection function, will pull the zillow dataset from the Codeup MySQL server. After that, it will also save a copy of 
    the csv locally if there wasn't one, so it doesn't have to run the query each time.
    '''
    if os.path.isfile('zillow_2017.csv'):
        return pd.read_csv('zillow_2017.csv')
    else:
        url = get_connection('zillow')
        query = '''
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
                       taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                JOIN propertylandusetype USING(propertylandusetypeid)
                WHERE propertylandusedesc = 'Single Family Residential';
                '''
        zillow = pd.read_sql(query, url)
        zillow.to_csv('zillow_2017.csv')
        return 

def wrangle_zillow(df):
    '''
    This function will, using an already existing DataFrame, clean the zillow_2017.csv created with my acquire function to make it ready to be split and explored.
    '''
    df = df.drop(columns='Unnamed: 0')
    df = df.dropna()
    
    df.bedroomcnt = df.bedroomcnt.astype(int)
    df.calculatedfinishedsquarefeet = df.calculatedfinishedsquarefeet.astype(int)
    df.taxvaluedollarcnt = df.taxvaluedollarcnt.astype(int)
    df.yearbuilt = df.yearbuilt.astype(int)
    df.fips = df.fips.astype(int)
    
    return df

def split_train_test(df, col):
    '''
    This function takes in a DataFrame, along with a target column (variable), and split it into train, validate, test subsets for our modeling phase.
    '''
    seed = 42
    train, val_test = train_test_split(df, train_size=.5, random_state=seed, stratify=df[col])
    validate, test = train_test_split(val_test, train_size=.6, random_state=seed, stratify=val_test[col])
    
    return train, validate, test

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