"""
Data Processing Module for Car Features and MSRP Prediction

This module contains functions for loading, cleaning, and preprocessing the car dataset.
"""

import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the car dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and renamed dataframe
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Rename columns for better readability
    df = df.rename(columns={
        "Engine HP": "HP", 
        "Engine Cylinders": "Cylinders", 
        "Transmission Type": "Transmission", 
        "Driven_Wheels": "Drive Mode",
        "highway MPG": "MPG-H", 
        "city mpg": "MPG-C", 
        "MSRP": "Price"
    })
    
    return df

def clean_data(df):
    """
    Clean the car dataset by handling missing values.
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # Drop rows with missing values
    df_clean = df.dropna()
    
    return df_clean

def create_features(df):
    """
    Create new features from existing ones.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    # Create a copy to avoid modifying the original
    df_transformed = df.copy()
    
    # Add vehicle age
    current_year = datetime.datetime.now().year
    df_transformed['Vehicle Age'] = current_year - df_transformed['Year']
    
    # Create average MPG feature
    df_transformed["Average_MPG"] = df_transformed[["MPG-H", "MPG-C"]].mean(axis=1)
    
    return df_transformed

def encode_categorical(df):
    """
    One-hot encode categorical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with encoded categorical features
    """
    # Categorical columns to encode
    categorical_columns = ["Transmission", "Drive Mode", "Make"]
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    return df_encoded

def prepare_data(df, target_col='Price', drop_cols=None):
    """
    Prepare data for model training by splitting features and target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
        drop_cols (list): Additional columns to drop
        
    Returns:
        tuple: X (features) and y (target) dataframes
    """
    if drop_cols is None:
        drop_cols = []
    
    # Ensure target_col is not in drop_cols
    if target_col in drop_cols:
        drop_cols.remove(target_col)
    
    # Create features (X) and target (y)
    X = df.drop([target_col] + drop_cols, axis=1)
    y = df[target_col]
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

def data_pipeline(file_path, drop_cols=None):
    """
    Complete data processing pipeline from loading to train-test split.
    
    Args:
        file_path (str): Path to the CSV file
        drop_cols (list): Columns to drop from the features
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessed_df
    """
    # Load and clean data
    df = load_data(file_path)
    df_clean = clean_data(df)
    
    # Create features and encode categorical variables
    df_featured = create_features(df_clean)
    df_processed = encode_categorical(df_featured)
    
    # Prepare features and target
    X, y = prepare_data(df_processed, drop_cols=drop_cols)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    return X_train, X_test, y_train, y_test, df_processed
