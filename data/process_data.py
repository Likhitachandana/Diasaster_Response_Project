import sys
import pandas as pd 
from sqlalchemy import create_engine


def loading_data(messages_filepath, categories_filepath):

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    categories = df['categories'].str.split(pat=';', expand=True)

    row = categories.iloc[0]

    get_name = lambda x: x[:-2]
    column_names = [get_name(x) for x in row]

    categories.columns = column_names

    for column in categories:
        categories[column] = categories[column].str[-1]
        
        categories[column] = pd.to_numeric(categories[column])
    categories['related'] = categories['related'].apply(lambda x : 1 if x == 2 else x)

    df.drop(columns=['categories'], inplace=True, axis=1)

    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('dataset', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading the data from CSV files\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = loading_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('')


if __name__ == '__main__':
    main()

