"""
Modeling Module for Car Features and MSRP Prediction

This module contains functions for building, training, and evaluating models
to predict car prices based on features.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        object: Trained linear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train a decision tree regressor model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        random_state (int): Random seed for reproducibility
        
    Returns:
        object: Trained decision tree model
    """
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a random forest regressor model.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        n_estimators (int): Number of trees in the forest
        random_state (int): Random seed for reproducibility
        
    Returns:
        object: Trained random forest model
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a regression model's performance.
    
    Args:
        model (object): Trained model
        X_test (pd.DataFrame): Testing features
        y_test (pd.Series): Testing target
        model_name (str): Name of the model for display purposes
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print(f"{model_name} Results:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}\n")
    
    return {
        "model_name": model_name,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "predictions": y_pred
    }

def calculate_feature_importance(model, feature_names, plot=True, figsize=(10, 6)):
    """
    Calculate and optionally plot feature importance.
    
    Args:
        model (object): Trained tree-based model with feature_importances_ attribute
        feature_names (list): List of feature names
        plot (bool): Whether to plot feature importance
        figsize (tuple): Figure size for the plot
        
    Returns:
        pd.Series: Feature importance series
    """
    # Get feature importance
    importance = model.feature_importances_
    
    # Create Series with feature names as index
    feature_importance = pd.Series(importance, index=feature_names).sort_values(ascending=False)
    
    # Plot feature importance if requested
    if plot:
        plt.figure(figsize=figsize)
        feature_importance.plot(kind='bar')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    
    return feature_importance

def save_model(model, model_path, feature_names=None):
    """
    Save a trained model to disk.
    
    Args:
        model (object): Trained model to save
        model_path (str): Path to save the model
        feature_names (list): List of feature names used by the model
        
    Returns:
        str: Path where the model was saved
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    model_info = {
        'model': model,
        'feature_names': feature_names
    }
    joblib.dump(model_info, model_path)
    
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        tuple: (model, feature_names)
    """
    # Load the model
    model_info = joblib.load(model_path)
    
    model = model_info['model']
    feature_names = model_info.get('feature_names')
    
    return model, feature_names

def create_submission(model, X_test, output_path="submission.csv"):
    """
    Create a submission file with predictions.
    
    Args:
        model (object): Trained model
        X_test (pd.DataFrame): Testing features
        output_path (str): Path to save the submission file
        
    Returns:
        pd.DataFrame: Submission dataframe
    """
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'Id': X_test.index,
        'Predicted_Price': predictions
    })
    
    # Save to CSV
    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
    
    return submission
