"""
Car Features and MSRP Prediction - Prediction Script

This script loads a trained model and makes predictions on new car data.

Usage:
    python predict.py --model_path=models/random_forest_model.pkl --data_path=data/test_cars.csv
"""

import argparse
import pandas as pd
import numpy as np
from src.modeling import load_model, create_submission
from src.data_processing import load_data, clean_data, create_features, encode_categorical, prepare_data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with trained car price model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model file')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the car data CSV file for prediction')
    parser.add_argument('--output_path', type=str, default='submission.csv',
                      help='Path to save the prediction results')
    return parser.parse_args()

def process_data_for_prediction(file_path, feature_names):
    """
    Process data for prediction, ensuring it matches the training features.
    
    Args:
        file_path (str): Path to the CSV file
        feature_names (list): List of feature names used by the model
        
    Returns:
        pd.DataFrame: Processed dataframe ready for prediction
    """
    # Load and preprocess data
    df = load_data(file_path)
    df_clean = clean_data(df)
    df_featured = create_features(df_clean)
    df_processed = encode_categorical(df_featured)
    
    # Check if all required features are present
    missing_features = [feat for feat in feature_names if feat not in df_processed.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with zeros
        for feat in missing_features:
            df_processed[feat] = 0
    
    # Select only the features needed for prediction
    X_pred = df_processed[feature_names]
    
    return X_pred, df_processed

def main():
    """Main function to load model and make predictions."""
    # Parse arguments
    args = parse_args()
    
    # Load the model
    print(f"Loading model from {args.model_path}...")
    model, feature_names = load_model(args.model_path)
    
    if feature_names is None:
        print("Error: Model doesn't have feature names information.")
        return
    
    # Process data
    print(f"Processing data from {args.data_path}...")
    X_pred, df_processed = process_data_for_prediction(args.data_path, feature_names)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_pred)
    
    # Create submission file
    print(f"Creating submission file at {args.output_path}...")
    submission = pd.DataFrame({
        'Id': X_pred.index,
        'Predicted_Price': predictions
    })
    
    # Save to CSV
    submission.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")
    
    # Display summary statistics
    print("\nPrediction Summary:")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Average predicted price: ${predictions.mean():.2f}")
    print(f"Min predicted price: ${predictions.min():.2f}")
    print(f"Max predicted price: ${predictions.max():.2f}")

if __name__ == "__main__":
    main()
