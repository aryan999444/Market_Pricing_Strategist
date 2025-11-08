import pandas as pd
import numpy as np
import re

def extract_features(df):
    """
    Applies comprehensive data cleaning and feature engineering logic 
    to prepare the raw data for ML models.
    """
    
    # --- 1. Data Cleaning and Imputation ---
    
    # Price Cleaning (as already defined)
    median_price = df['Price'].median()
    df['Price'].fillna(median_price, inplace=True)
    
    # FIX: Impute missing Rating/Review_Count with median (CRUCIAL for model training)
    if 'Rating' in df.columns and df['Rating'].isnull().any():
        df['Rating'].fillna(df['Rating'].median(), inplace=True)
    if 'Review_Count' in df.columns and df['Review_Count'].isnull().any():
        df['Review_Count'].fillna(df['Review_Count'].median(), inplace=True)
        
    # FIX: Create binary availability feature (was missing)
    if 'Availability' in df.columns:
        df['Is_Available'] = df['Availability'].apply(
            lambda x: 1 if 'in stock' in str(x).lower() else 0
        )
    else:
        # Default to 1 (available) if column is missing, to keep feature set consistent
        df['Is_Available'] = 1 


    # --- 2. Feature Engineering from Product_Name ---
    
    # Extract Brand (Confirmed: uses correct 'Product_Name' case)
    df['Brand'] = df['Product_Name'].apply(lambda x: str(x).split(' ')[0])
    
    # Helper function for Storage extraction
    def extract_storage(name):
        name = str(name).upper()
        match = re.search(r'(\d+)\s*(GB|TB)', name)
        if match:
            return match.group(0).replace(' ', '')
        return 'Unknown'
        
    # Extract Storage
    df['Storage_Capacity'] = df['Product_Name'].apply(extract_storage)
    
    # --- 3. Binary Indicator Features ---
    
    df['Has_Noise_Cancelling'] = df['Product_Name'].str.contains(r'Noise-?Cancelling|ANC', case=False, na=False, regex=True).astype(int)
    df['Is_Wireless'] = df['Product_Name'].str.contains(r'Wireless|Bluetooth|Cordless', case=False, na=False, regex=True).astype(int)
    df['Supports_5G'] = df['Product_Name'].str.contains(r'5G', case=False, na=False, regex=True).astype(int)
    df['Is_Ultra_HD'] = df['Product_Name'].str.contains(r'Ultra HD|4K|8K|UHD', case=False, na=False, regex=True).astype(int)
    df['Is_Smart_TV'] = df['Product_Name'].str.contains(r'Smart TV|SmartTV|Android TV|WebOS|Tizen|Roku', case=False, na=False, regex=True).astype(int)
    df['Is_LED'] = df['Product_Name'].str.contains(r'LED|OLED|QLED', case=False, na=False, regex=True).astype(int) 
    df['Is_Gaming'] = df['Product_Name'].str.contains(r'Gaming|RGB|Console|PlayStation|Xbox|Nintendo', case=False, na=False, regex=True).astype(int) 
    df['Is_Pro_Model'] = df['Product_Name'].str.contains(r'Pro|Professional|Premium', case=False, na=False, regex=True).astype(int) 

    # --- 4. Target Definition (Need for training) ---
    RATING_THRESHOLD = 4.5
    REVIEW_COUNT_THRESHOLD = 1000
    
    df['Is_High_Performer'] = np.where(
        (df['Rating'] >= RATING_THRESHOLD) & (df['Review_Count'] >= REVIEW_COUNT_THRESHOLD),1,0
    )
    
    return df