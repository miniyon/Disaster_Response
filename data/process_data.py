import sqlite3
import pandas as pd
import sys


def read_data(filename):
    
    """This function reads an input file of the csv/xls/xlsx types

    Args:
        filename (string): input file

    Returns:
        dataframe: the input file loaded into a dataframe
    """    
    ext = filename.split('.')[1]
    if ext == 'csv':
        df = pd.read_csv(filename)
    elif ( ext == 'xlsx') | (ext == 'xls'):
        df = pd.read_excel(filename)
    return df



def merge_data(message,cleaned,id_):
    
    """This function merges two dataframes on a common column;
    drops duplicates across all columns

    Args:
        message (dataframe): the input message dataframe
        cleaned (dataframe): the cleaned dataframe

    Returns:
        dataframe: merged dataframe without duplicates on all columns
    """    

    data = pd.merge(message,cleaned,on=id_)
    data.drop_duplicates(inplace=True)
    return data


def clean_data(message,target,target_col,id_,genre_dict):

    """this function cleans the target 
       dataframe by converting the single column to multiple columns
       with the associated values as column values

       Args:
        message (dataframe): the input message dataframe
        target (dataframe): the input target dataframe

    Returns:
        dataframe: cleaned and merged dataframe
    """    

    items = []
    for item in target[target_col]:
        items_list = item.split(';')
        items.append(items_list)
    x = pd.DataFrame(items)
    x_temp = pd.concat([target[id_],x],axis=1)
    columns = []
    for col in x_temp.columns[1:]:
        column_name = [str(x).split('-')[0] for x in x_temp[col].unique().tolist()][0]
        columns.append(column_name)
        x_temp[col] = x_temp[col].apply(lambda x : str(x).split('-')[1])  
    x_temp.columns = [id_] + columns
    df_final = pd.merge(message[['id','message','genre']],x_temp,on='id')
    
    df_final['genre'] = df_final['genre'].apply(lambda x : genre_dict[x])
    df_final.drop_duplicates(inplace=True)
    return df_final


def insert_into_db(db_name,table_name,df):
    
    """this function inserts the processed data 
    into the database

    Args:
        db_name (string): DB name into which the records have to inserted
        table_name (string): table name for data insertion
        df (dataframe): data that has to be inserted
    """    
    try:
        conn = sqlite3.connect(db_name)
        df.to_sql(table_name, con = conn,if_exists='replace', index=False)
    except:
        raise
    finally:
        conn.close()
        
def main():
    if len(sys.argv) == 4:
        target_col,id_col = 'categories','id'
        table_name = 'disaster_response_tweets'
        messages,categories,db_name = sys.argv[1:]
        genre_dict = {'news':0,'direct':1,'social':2}
        
        print('Reading the input data...\n    FileName: {}'.format(messages))
        data = read_data(messages)
        
        print('Reading the input data...\n    FileName: {}'.format(categories))
        categories = read_data(categories)

        print('Cleaning the data ...')
        data = clean_data(data,categories,target_col,id_col,genre_dict)

        print('Inserting into Database...\n    Database: {} , Table : {}'.format(db_name,table_name))
        insert_into_db(db_name,table_name,data)

        print('Cleaned Data saved!')

    else:
        print('Please provide the filepath of the messages '\
              'as the first argument and the filepath of the labels file '\
              'as the second argument. ' \
              'and the Database name as the third argument')


if __name__ == '__main__':
    main()