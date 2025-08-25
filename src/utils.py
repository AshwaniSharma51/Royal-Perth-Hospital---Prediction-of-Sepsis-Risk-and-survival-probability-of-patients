from pathlib import Path
import pandas as pd
import numpy as np

import missingno as msno
import matplotlib.pyplot as plt

def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent

def save_csv(df, filename):
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    
        
def categorise_age(df):
    """
    Categorise the 'AGE' column in the DataFrame into age bins.
    
    Parameters:
    - df: DataFrame containing the 'AGE' column
    
    Returns:
    - DataFrame with an additional 'AGE_cat' column containing the age categories
    """

    # define age bin edges
    age_bin_edges = [0, 1, 18, 36, 51, 71, 121]
    
    # define bin labels
    age_bin_labels = ['0', '1-17', '18-35', '36-50', '51-70', '71+']
    
    # create age bins
    df['AGE_cat'] = pd.cut(df['AGE'], bins=age_bin_edges, labels=age_bin_labels, right=False)
    
    return df

def get_sepsis_codes(df_desc_icd):
    """
    Get ICD-9 codes relating to sepsis
    
    Parameters:
    
    Returns:
    
    """
    
    icd_sepsis = df_desc_icd[df_desc_icd.apply(lambda x:'sepsis' in x['SHORT_TITLE'].lower(),axis=1)]['ICD9_CODE'].values
    
    return icd_sepsis

def get_sepsis_admissions(icd_sepsis, df):
    """

    Args:
        icd_sepsis (list): List of ICD-9 codes relating to sepsis.
        df (DataFrame): DataFrame containing column ICD9_CODE.
    
    Returns:
        df_sepsis_admissions (DataFrame): DataFrame containing sepsis admissions.
    """

    df_sepsis_admissions = df[df.apply(lambda x:x['ICD9_CODE'] in icd_sepsis, axis=1)]
    
    return df_sepsis_admissions
    
def filter_missing_rows(df, proportion_to_return):
    """
    Filters rows in a DataFrame based on a threshold of missing or masked values.

    This function checks each row in the input DataFrame for missing or masked values. 
    Missing values are identified as NaN, while masked values are represented as -999. 
    The function calculates the percentage of missing or masked values in each row. 
    Rows with a percentage of missing or masked values less than or equal to the specified 
    'proportion_to_return' are retained in the returned DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with potential missing or masked values.
    proportion_to_return (float): The threshold percentage (0-100) of allowable missing or 
                                   masked values per row. Rows with a percentage of missing 
                                   or masked values within this threshold are returned.

    Returns:
    pd.DataFrame: A DataFrame consisting of rows from the input DataFrame that have a 
                  percentage of missing or masked values less than or equal to the 
                  'proportion_to_return'. If no masking is present, the function checks 
                  for NaN values instead.
    """
    # Check if masking has been applied
    if (df == -999).any().any():  # Masking is present
        count_missing = (df == -999).sum(axis=1)
    else:  # No masking, check for NaN
        count_missing = df.isna().sum(axis=1)

    # Calculate the percentage of missing values for each row
    p_missing = count_missing / df.shape[1] * 100

    # Filter rows based on the allowed proportion of missing/masked values
    return df[p_missing <= proportion_to_return]

def visualise_missing_data(df):
    """
    Convert non-standard missing values (-999) to NaN and visualize the missing data using missingno,
    without altering the original DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        None
    """

    # Create a copy of the DataFrame to avoid changing the original
    df_copy = df.copy()

    # Replace -999 with NaN in the copy
    df_copy.replace(-999, np.nan, inplace=True)

    plt.figure(figsize=(15, 7)) 

    # Create a matrix plot for missing values
    msno.matrix(df_copy)
    plt.show()

    # Create a bar plot for missing values
    msno.bar(df_copy)
    plt.show()

    # Create a heatmap for missing values
    msno.heatmap(df_copy)
    plt.show()