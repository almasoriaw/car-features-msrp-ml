"""
Visualization Module for Car Features and MSRP Prediction

This module contains functions for visualizing exploratory data analysis,
feature relationships, and model results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def set_plotting_style():
    """Set the default Seaborn style for consistent visualizations."""
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_correlation_heatmap(df, figsize=(12, 10)):
    """
    Plot a correlation heatmap for numerical features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        figsize (tuple): Figure size for the plot
    """
    # Calculate correlation matrix for numeric columns only
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        annot=True, 
        fmt='.2f', 
        square=True, 
        linewidths=.5
    )
    
    plt.title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_feature_distributions(df, features, figsize=(18, 12)):
    """
    Plot distributions of specified features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of feature names to plot
        figsize (tuple): Figure size for the plot
    """
    # Determine grid dimensions
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each feature
    for i, feature in enumerate(features):
        if i < len(axes):
            sns.histplot(df[feature], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_scatter_with_price(df, features, figsize=(18, 12)):
    """
    Plot scatter plots of features against price.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of feature names to plot against price
        figsize (tuple): Figure size for the plot
    """
    # Determine grid dimensions
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each feature against price
    for i, feature in enumerate(features):
        if i < len(axes):
            sns.scatterplot(x=feature, y='Price', data=df, alpha=0.6, ax=axes[i])
            axes[i].set_title(f'{feature} vs Price')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Price')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, features, figsize=(18, 12)):
    """
    Plot boxplots for specified features to identify outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        features (list): List of feature names to plot
        figsize (tuple): Figure size for the plot
    """
    # Determine grid dimensions
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each feature
    for i, feature in enumerate(features):
        if i < len(axes):
            sns.boxplot(x=df[feature], ax=axes[i])
            axes[i].set_title(f'Boxplot of {feature}')
            axes[i].set_xlabel(feature)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_categorical_analysis(df, cat_features, target='Price', figsize=(18, 12)):
    """
    Plot the relationship between categorical features and the target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        cat_features (list): List of categorical feature names
        target (str): Target variable name
        figsize (tuple): Figure size for the plot
    """
    # Determine grid dimensions
    n_features = len(cat_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each categorical feature
    for i, feature in enumerate(cat_features):
        if i < len(axes):
            # Limit to top categories if there are many
            value_counts = df[feature].value_counts()
            if len(value_counts) > 15:
                top_cats = value_counts.nlargest(15).index
                data = df[df[feature].isin(top_cats)].copy()
                title_suffix = " (top 15 categories)"
            else:
                data = df.copy()
                title_suffix = ""
            
            sns.boxplot(x=feature, y=target, data=data, ax=axes[i])
            axes[i].set_title(f'{feature} vs {target}{title_suffix}')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel(target)
            plt.setp(axes[i].get_xticklabels(), rotation=45, ha='right')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_prediction_vs_actual(y_true, y_pred, title="Predicted vs Actual Values", figsize=(10, 6)):
    """
    Plot predicted values against actual values.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        title (str): Plot title
        figsize (tuple): Figure size for the plot
    """
    plt.figure(figsize=figsize)
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title(title)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, figsize=(10, 6)):
    """
    Plot residuals of predictions.
    
    Args:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values
        figsize (tuple): Figure size for the plot
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=figsize)
    
    # Residuals vs Predicted scatter plot
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Residuals distribution
    plt.figure(figsize=figsize)
    sns.histplot(residuals, kde=True)
    plt.title('Residuals Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_transformation(df, feature, transformations, figsize=(18, 6)):
    """
    Plot multiple transformations of a feature against the target.
    
    Args:
        df (pd.DataFrame): Input dataframe
        feature (str): Feature name to transform
        transformations (dict): Dictionary mapping transformation names to functions
        figsize (tuple): Figure size for the plot
    """
    n_transforms = len(transformations)
    
    # Create figure
    fig, axes = plt.subplots(1, n_transforms + 1, figsize=figsize)
    
    # Plot original feature
    sns.scatterplot(x=feature, y='Price', data=df, alpha=0.6, ax=axes[0])
    axes[0].set_title(f'Original: {feature} vs Price')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Price')
    
    # Plot each transformation
    for i, (name, func) in enumerate(transformations.items(), 1):
        # Apply transformation
        transformed_feature = f"{name}_{feature}"
        transformed_values = func(df[feature])
        
        # Create scatter plot
        sns.scatterplot(x=transformed_values, y=df['Price'], alpha=0.6, ax=axes[i])
        axes[i].set_title(f'{name}: {transformed_feature} vs Price')
        axes[i].set_xlabel(transformed_feature)
        axes[i].set_ylabel('Price')
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results_dict, metric='r2', figsize=(10, 6)):
    """
    Plot comparison of model performances.
    
    Args:
        results_dict (dict): Dictionary mapping model names to their evaluation results
        metric (str): Metric to compare ('r2', 'mse', or 'rmse')
        figsize (tuple): Figure size for the plot
    """
    # Extract model names and metrics
    models = list(results_dict.keys())
    
    if metric == 'r2':
        values = [results_dict[model]['r2'] for model in models]
        title = 'Model Comparison - R² Score (higher is better)'
    elif metric == 'mse':
        values = [results_dict[model]['mse'] for model in models]
        title = 'Model Comparison - MSE (lower is better)'
    elif metric == 'rmse':
        values = [results_dict[model]['rmse'] for model in models]
        title = 'Model Comparison - RMSE (lower is better)'
    else:
        raise ValueError("metric must be one of: 'r2', 'mse', 'rmse'")
    
    # Create bar plot
    plt.figure(figsize=figsize)
    bars = plt.bar(models, values)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel(metric.upper())
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
