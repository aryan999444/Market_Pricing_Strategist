import pandas as pd
import numpy as np
import re

def extract_features(df):
    """Applies feature engineering logic"""
    # Price Cleaning
    median_price = df['Price'].median()
    df['Price'].fillna(median_price, inplace=True)
    # Extract Brand here
    df['Brand'] = df['Product_name'].apply(lambda x: str(x).split(' ')[0])
    # Extract Storage
    def extract_storage(name):
        name = str(name).upper()
        match = re.search(r'(\d+)\s*(GB|TB)', name)
        if match:
            return match.group(0).replace(' ', '')
        return 'Unknown'
    df['Storage_Capacity'] = df['Product_Name'].apply(extract_storage)
    # 4. Indicator Features (The code you provided)
    df['Has_Noise_Cancelling'] = df['Product_Name'].str.contains(r'Noise-?Cancelling|ANC', case=False, na=False, regex=True).astype(int)
    df['Is_Wireless'] = df['Product_Name'].str.contains(r'Wireless|Bluetooth|Cordless', case=False, na=False, regex=True).astype(int)
    df['Supports_5G'] = df['Product_Name'].str.contains(r'5G', case=False, na=False, regex=True).astype(int)
    df['Is_Ultra_HD'] = df['Product_Name'].str.contains(r'Ultra HD|4K|8K|UHD', case=False, na=False, regex=True).astype(int)
    df['Is_Smart_TV'] = df['Product_Name'].str.contains(r'Smart TV|SmartTV|Android TV|WebOS|Tizen|Roku', case=False, na=False, regex=True).astype(int)
    # Target Defination (Need for training)
    RATING_THRESHOLD = 4.5
    REVIEW_COUNT_THRESHOLD = 1000
    df['Is_High_Performer'] = np.where(
        (df['Rating'] >= RATING_THRESHOLD) & (df['Review_Count'] >= REVIEW_COUNT_THRESHOLD),1,0
    )
    return df
