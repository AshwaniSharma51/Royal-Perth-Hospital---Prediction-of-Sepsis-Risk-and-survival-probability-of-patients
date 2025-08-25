import os
import dask.dataframe as dd
import pandas as pd
from datetime import datetime

def load_labevents_dask(file_dir):
    print(str(datetime.now()) + ' Start')

    # Read all Parquet files and concatenate the list of DataFrames into a single DataFrame
    ddf = dd.read_parquet(path=f'{file_dir}/*.parquet',
                        index='row_key',
                        engine='fastparquet')

    print(str(datetime.now()) + ' ' + str(len(ddf)))
    
    ddf.columns = ddf.columns.str.upper()

    # Get OutOfMemory crash when converting to pandas
    # df = ddf.compute()

    print(str(datetime.now()) + ' End')
    return ddf

def load_labevents_pd(dir):
    # List all Parquet files in the output folder
    parquet_files = sorted([f'{dir}/{file}' for file in os.listdir(dir) if file.endswith('.parquet')])

    print(str(datetime.now()) + ' Start')

    # Read all Parquet files and concatenate the list of DataFrames into a single DataFrame
    df = None
    for file in parquet_files:
        df_i = pd.read_parquet(file, 'fastparquet')
        # print(str(datetime.now()) + ' ' + file)
        if df is None:
            df = df_i
        else:
            df = pd.concat([df, df_i], ignore_index = True)
    
    df.columns = df.columns.str.upper()

    print(str(datetime.now()) + ' ' + str(len(df)))

    print(str(datetime.now()) + ' End')
    return df