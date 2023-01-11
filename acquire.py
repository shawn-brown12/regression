import os
import pandas as pd
from env import host, username, password
#this will connect to Codeup mysql server if this function is within the env.py file in directory


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