"""
Car Features and MSRP Prediction - Model Training Script

This script trains models to predict car prices based on features
and saves the best performing model.

Usage:
    python train_model.py --data_path=data/cars_data.csv
"""

import argparse
import os
import pandas as pd
import numpy as np
from src.data_processing import data_pipeline
from src.modeling import (
    train_linear_regression, 
    train_decision_tree,
    train_random_forest,
    evaluate_model,
    calculate_feature_importance,
    save_model
)
from src.visualization import (
    set_plotting_style,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_scatter_with_price,
    plot_prediction_vs_actual,
    plot_model_comparison
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train car price prediction models')
    parser.add_argument('--data_path', type=str, default='data/cars_data.csv',
                      help='Path to the car dataset CSV file')
    parser.add_argument('--model_dir', type=str, default='models',
                      help='Directory to save trained models')
    parser.add_argument('--random_state', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true',
                      help='Create and display visualizations')
    return parser.parse_args()

def main():
    """Main function to train and evaluate models."""
    # Parse arguments
    args = parse_args()
    
    # Set plotting style
    set_plotting_style()
    
    print("Loading and preprocessing data...")
    # Process data through pipeline
    X_train, X_test, y_train, y_test, df_processed = data_pipeline(
        args.data_path, 
        drop_cols=['Model']  # Drop columns that aren't useful for modeling
    )
    
    # Display basic data information
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Create visualizations if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        
        # Select numerical features
        numerical_features = df_processed.select_dtypes(include=['float64', 'int64']).columns
        features_to_plot = [col for col in numerical_features if col != 'Price'][:6]  # Limit to 6 features
        
        # Create visualizations
        plot_correlation_heatmap(df_processed[numerical_features])
        plot_feature_distributions(df_processed, features_to_plot)
        plot_scatter_with_price(df_processed, features_to_plot)
    
    # Train models
    print("\nTraining models...")
    
    # Linear Regression
    print("Training Linear Regression...")
    lr_model = train_linear_regression(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Linear Regression")
    
    # Decision Tree
    print("Training Decision Tree...")
    dt_model = train_decision_tree(X_train, y_train, random_state=args.random_state)
    dt_results = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train, random_state=args.random_state)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Compare models
    results = {
        "Linear Regression": lr_results,
        "Decision Tree": dt_results,
        "Random Forest": rf_results
    }
    
    if args.visualize:
        plot_model_comparison(results, metric='r2')
        plot_model_comparison(results, metric='rmse')
    
    # Find best model based on R²
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree": dt_model,
        "Random Forest": rf_model
    }[best_model_name]
    
    print(f"\nBest model: {best_model_name} with R² of {results[best_model_name]['r2']:.2f}")
    
    # Calculate feature importance for the best model if it's a tree-based model
    if best_model_name in ["Decision Tree", "Random Forest"]:
        print("\nCalculating feature importance...")
        feature_importance = calculate_feature_importance(
            best_model, X_train.columns, plot=args.visualize
        )
        print("Top 5 most important features:")
        print(feature_importance.head())
    
    # Plot actual vs predicted values for the best model
    if args.visualize:
        best_predictions = results[best_model_name]['predictions']
        plot_prediction_vs_actual(y_test, best_predictions, 
                                 f"{best_model_name}: Predicted vs Actual Prices")
    
    # Save the best model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, f"{best_model_name.lower().replace(' ', '_')}_model.pkl")
    save_model(best_model, model_path, feature_names=list(X_train.columns))
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()
