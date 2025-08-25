import numpy as np
import pandas as pd
from pathlib import Path


# Requires the following from Chartevents:
    # MAP (Mean arterial bloodpressure) 
        # CHARTEVENTS['ITEMID'] == [220052,225312,224322]

    # Heart rate
        # CHARTEVENTS['ITEMID'] == [211, 220045]

    # SBP (systolic blood pressure) (arterial blood pressure (systolic))
        # CHARTEVENTS['ITEMID'] == [51, 220050, 225309]

    # Respiratory Rate
        # CHARTEVENTS['ITEMID'] == [618, 220210, 3603, 224689, 614, 651, 224422, 615, 224690]




def get_qsofa_score(df):
    """
    Parameters: 
    - df [CHARTEVENTS] containing Resp and SBP
    
    Returns: 
    - qsofa score. 1 point for each of the following:
        RR > 22/min
        SBP <= 100mmHg
        Altered mental status    # Requires GCS, which we don't have access to
    """
    qsofa = pd.DataFrame(index=df.index)
    
    # check if in range
    qsofa['Resp'] = df['Resp'] >= 22
    qsofa['SBP'] = df['SBP'] <= 100
    
    # check for null
    qsofa['Resp'][df['Resp'] == -999] = -999
    qsofa['SBP'][df['SBP'] == -999] = -999
    
    # sum values
    qsofa_score = qsofa.sum(axis=1)
    
    # check if sum is less than 0, if so set it to null (-999)
    qsofa_score[qsofa_score < 0] = -999

    return qsofa_score



def get_sofa_score(df):
    """
    Refer to https://www.mdcalc.com/sequential-organ-failure-assessment-sofa-score#evidence
    
    Parameters:
    - df: A DataFrame containing: 
        Platelet_count [51265]
        Bilirubin_total [50885]
        Creatinine [50912]
        Mean Arterial Pressure     # Requires Chart Events
    
    
    Returns:
    - SOFA score.
    """
    sofa = pd.DataFrame(index=df.index)
    ITEMID_51265 = 'ITEMID_51265'
    sofa['Platelets'] = 0
    sofa['Platelets'][(400 <= df[ITEMID_51265]) & (df[ITEMID_51265] < np.inf)] = 0
    sofa['Platelets'][(300 <= df[ITEMID_51265]) & (df[ITEMID_51265] < 400)] = 1
    sofa['Platelets'][(200 <= df[ITEMID_51265]) & (df[ITEMID_51265] < 300)] = 2
    sofa['Platelets'][(100 <= df[ITEMID_51265]) & (df[ITEMID_51265] < 200)] = 3
    sofa['Platelets'][df[ITEMID_51265] < 100] = 4
    sofa['Platelets'][df[ITEMID_51265] == -999] = -999     # Set the value to 0 for rows == -999

    ITEMID_50885 = 'ITEMID_50885'
    sofa['Bilirubin'] = 0
    sofa['Bilirubin'][df[ITEMID_50885] < 1.2] = 0
    sofa['Bilirubin'][(1.2 <= df[ITEMID_50885]) & ((df[ITEMID_50885]) <= 1.9)] = 1
    sofa['Bilirubin'][(1.9 < df[ITEMID_50885]) & ((df[ITEMID_50885]) <= 5.9)] = 2
    sofa['Bilirubin'][(5.9 < df[ITEMID_50885]) & ((df[ITEMID_50885]) <= 11.9)] = 3
    sofa['Bilirubin'][(11.9 < df[ITEMID_50885])] = 4
    sofa['Bilirubin'][df[ITEMID_50885] == -999] = -999     # Set the value to 0 for rows == -999

    ITEMID_50912 = 'ITEMID_50912'
    sofa['Creatinine'] = 0
    sofa['Creatinine'][df[ITEMID_50912] < 1.2] = 0
    sofa['Creatinine'][(1.2 <= df[ITEMID_50912]) & (df[ITEMID_50912] < 1.9)] = 1
    sofa['Creatinine'][(1.9 <= df[ITEMID_50912]) & (df[ITEMID_50912] < 3.5)] = 2
    sofa['Creatinine'][(3.5 <= df[ITEMID_50912]) & (df[ITEMID_50912] < 5)] = 3
    sofa['Creatinine'][(df[ITEMID_50912] >= 5)] = 4
    sofa['Creatinine'][df[ITEMID_50912] == -999] = -999     # Set the value to 0 for rows == -999

    # todo: revise when chartevents are available
    # sofa['MAP'][df['MAP'] < 70] = 1
    
    sofa_score = sofa.sum(axis=1)
    
    # Check if the sum is less than 0, if so set it to -999
    sofa_score[sofa_score < 0] = -999
    
    return sofa_score

    

    
def get_sirs_score(df):
    """
    Evaluate to see if they satisfy any 2 SIRS criteria:
        1. Temp > 38degC  OR  Temp < 36degC
        2. HR > 90
        3. RespRate > 20  OR  PaCO2 < 32mmHg
        4. WBC > 12,000/mm^3  OR  WBC < 4,000/mm^3  OR  > 10% bands
        
    Parameters:
    - dataframe with: Temp, HR, Resp, PaCO2, WBC
    
    Output:
    - SIRS score
    """
    # Create a dataframe that stores true false for each category
    df_sirs = pd.DataFrame(index=df.index, columns=['temp', 'hr', 'rr.paco2', 'wbc'])
    df_sirs['temp'] = ((df['Temp'] > 38) | (df['Temp'] < 36))
    df_sirs['hr'] = df['HR'] > 90
    df_sirs['rr.paco2'] = ((df['Resp'] > 20) | (df['PaCO2'] < 32))
    df_sirs['wbc'] = ((df['WBC'] < 4) | (df['WBC'] > 12))

    # Sum each row, if >= 2, then mark as SIRS
    df_sum = df_sirs.sum(axis=1) >= 2

    # Add SIRS to df
    df['SIRS'] = df_sum

    return df