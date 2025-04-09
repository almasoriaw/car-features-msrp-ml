"""
Data Download Script for Car Features and MSRP Prediction

This script downloads the car dataset from Google Drive and saves it to the data directory.

Usage:
    python download_data.py
"""

import os
import argparse
import pandas as pd
try:
    import gdown
except ImportError:
    print("gdown not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "gdown"])
    import gdown

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download car dataset')
    parser.add_argument('--output_dir', type=str, default='data',
                      help='Directory to save the dataset')
    parser.add_argument('--filename', type=str, default='cars_data.csv',
                      help='Name of the output file')
    return parser.parse_args()

def main():
    """Main function to download the dataset."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Google Drive file ID (Extracted from Google Drive sharable link)
    file_id = "1H7cbu0NiqUFViY6IOtomNgDn2AzPzEtS"
    output_path = os.path.join(args.output_dir, args.filename)
    
    print(f"Downloading car dataset to {output_path}...")
    
    # Download the file using gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    
    # Verify the download
    try:
        df = pd.read_csv(output_path)
        print(f"Dataset downloaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        print(f"Column names: {', '.join(df.columns)}")
    except Exception as e:
        print(f"Error verifying the dataset: {e}")
    
    print("\nDownload complete!")

if __name__ == "__main__":
    main()
