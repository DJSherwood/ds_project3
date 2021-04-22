import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
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
    
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('DisasterResponse', engine, index=False)  


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