import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    '''
    Creates a dataframe by merging the message and category data together.
    INTPUT: message data location (string) and category data location (string)
    OUTPUT: pandas dataframe
    '''
    # load messages into dataframe
    messages = pd.read_csv(messages_filepath)
    # load categories into dataframe
    categories = pd.read_csv(categories_filepath)
    # merge messages and categories
    df = pd.merge(
        left=categories,
        right=messages,
        how="left",
        on="id"
    )
    df = df.reset_index(drop=True)
    
    return df


def clean_data(df):
    '''
    Cleans the dataframe. 
    INPUT: pandas dataframe
    OUTPUT: pandas dataframe
    '''
    # abbreviate category names
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_names = row.apply(lambda x: x[:-2])
    # overwrite the column names with abbreviated categories
    categories.columns = category_names
    # convert column from string to numeric
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(-1)
        # convert column from string to numeric
        categories[column] = categories[column].astype("int64")
    
    # drop the original categories column from `df`
    df = df.drop(columns='categories')
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    # convert all indicator columns to integer
    df = df.astype({ col_name : 'int64' for col_name in df.columns[4:]})
    # the 'relevant' category has values of '2' for some reason
    df = df.replace({'related' : 2}, 1)
    
    return df

def save_data(df, database_filename):
    ''' 
    Create a database to store the pandas datframe into.
    INPUT: pandas dataframe
    OUTPUT: sqlite3 database ( default table name is DisasterResponse ) 
    '''
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
