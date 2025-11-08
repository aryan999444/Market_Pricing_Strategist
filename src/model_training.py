import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from.utils import save_model, load_model

# --- Congiguration ---
TOP_N_BRANDS = 15
K_CLUSTER = 5
RANDOM_SEED = 42

def train_and_save_models(data_path='../data/processed_data.csv'):
    """
    Loads processed data, scales/encodes features, trains Clustering and 
    Classification models, and saves all assets (scaler, models).
    """
    print("--- Starting Automated Model Retraining ---")
    # Load data and Define Target
    try:
        df_final = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. PLease run data_Cleaning first.")
        return
    
    # Define features used in the model
    numerical_features = ['Price', 'Rating', 'Review_Count', 'Is_Available',
                          'Has_Noise_Cancelling', 'Is_Wireless', 'Supports_5G', 
                          'Is_Ultra_HD', 'Is_Smart_TV', 'Is_LED', 'Is_Gaming', 'Is_Pro_Model']

    categorical_features = ['Brand', 'Storage_Capacity']