# 🚗 Car Features and MSRP Prediction

![GitHub stars](https://img.shields.io/github/stars/almasoriaw/car-features-msrp-ml?style=social)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 📊 Project Overview

This project uses machine learning to predict car prices (MSRP - Manufacturer's Suggested Retail Price) based on various vehicle features such as make, model, horsepower, year, and other specifications. The analysis identifies which features most significantly impact car pricing.

## 🔍 Key Findings

- **Horsepower** is the most influential feature in determining car prices (67% importance)
- **Vehicle age** and **manufacturing year** together account for ~24% of price variability
- **Decision Tree** and **Random Forest** models significantly outperform Linear Regression
- Models achieve **R² scores up to 0.91**, indicating strong predictive performance

## 🚀 Features

- **Data Processing Pipeline**: Comprehensive data cleaning, feature engineering, and preprocessing
- **Multiple ML Models**: Implementation of Linear Regression, Decision Tree, and Random Forest models
- **Feature Importance Analysis**: Identification of key price determinants
- **Visualization Suite**: Rich set of visualizations for data exploration and model evaluation
- **Command-line Interface**: Easy-to-use scripts for training and prediction

## 🛠️ Technical Details

### Project Structure

```
car-features-msrp-ml/
├── .github/                    # GitHub-specific files
│   └── ISSUE_TEMPLATE/         # Issue templates
├── data/                       # Data directory
├── images/                     # Project images for documentation
├── models/                     # Saved model files
├── notebooks/                  # Jupyter notebooks
│   └── car_features_msrp_original.ipynb  # Original analysis
├── src/                        # Source code
│   ├── __init__.py
│   ├── data_processing.py      # Data preprocessing functions
│   ├── modeling.py             # Model training and evaluation
│   └── visualization.py        # Visualization utilities
├── .gitignore                  # Git ignore file
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
├── predict.py                  # Prediction script
├── requirements.txt            # Dependencies
└── train_model.py              # Model training script
```

### Data

The dataset contains information on various car models including:

- Make and model
- Year of production
- Engine specifications (horsepower, cylinders)
- Transmission type
- Drive mode
- Fuel efficiency (highway and city MPG)
- Prices (MSRP)

Data source: [Kaggle - Car Dataset](https://www.kaggle.com/datasets/CooperUnion/cardataset)

## 📊 Model Performance

| Model | Mean Squared Error (MSE) | Root Mean Squared Error (RMSE) | R² Score |
|-------|--------------------------|--------------------------------|----------|
| Linear Regression | 71,039,210 | 8,428.48 | 0.67 |
| Decision Tree | 25,824,534 | 5,082.77 | 0.88 |
| Random Forest | 18,257,490 | 4,272.88 | 0.91 |

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/almasoriaw/car-features-msrp-ml.git
   cd car-features-msrp-ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   - Option 1: Download from [Kaggle](https://www.kaggle.com/datasets/CooperUnion/cardataset)
   - Option 2: Use the data download script (requires gdown):
     ```bash
     python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1H7cbu0NiqUFViY6IOtomNgDn2AzPzEtS', 'data/cars_data.csv')"
     ```

### Usage

#### Training a Model

```bash
python train_model.py --data_path=data/cars_data.csv --visualize
```

#### Making Predictions

```bash
python predict.py --model_path=models/random_forest_model.pkl --data_path=data/test_cars.csv
```

## 📈 Visualizations

The project includes multiple visualizations:

- Feature distribution analysis
- Correlation heatmaps
- Scatter plots of features vs. price
- Feature importance charts
- Prediction vs. actual comparisons
- Residual analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- Dataset provided by Cooper Union via Kaggle
- Inspired by automotive market analysis techniques
- Built as part of Machine Learning Analyst Diploma program
