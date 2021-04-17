## This is the ETL Pipeline
# import libraries
from sqlalchemy import create_engine
import pandas as pd
# load messages dataset
messages = pd.read_csv("./messages.csv")
# load categories dataset
categories = pd.read_csv("./categories.csv")
# merge datasets
df = pd.merge(
    left = categories,
    right = messages,
    how = "left", 
    on = "id"
) 
df = df.reset_index(drop = True)
# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(':', expand = True)
# select the first row of the categories dataframe
row = categories.iloc[0]
# extract a list of new column names for categories
category_columns = row.apply(lambda x: x[:-2])
# rename the columns of 'categories'
categories.columns = category_colnames
# convert category values to just numbers 0 or 1
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str.slice(-1)

    # convert colum from string to numeric
    categories[column] = categories[column].astype("int64")

# drop the original categoires column from 'df'
df = df.drop(columns='categories')
# caoncatenate the original dataframe with the new categiores dataframe
df = pd.concat([df, categories], axis=1)
# check number of duplicates
df = df.drop_duplicates()
# save this to the sql engine
engine = create_engine('sqlite:///Messages.db')
df.to_sql('Messages.db', engine, index=False)
