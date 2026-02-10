# 🏠HOUSEWISE: A Real Estate Price Predictor Using Machine Learning

A machine learning project that predicts house prices based on property features using Random Forest Regressor.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 📊 Project Overview

This project uses machine learning to predict house sale prices in King County, Seattle. By analyzing property features such as number of bedrooms, bathrooms, square footage, and more, the model provides accurate price estimates.

**Key Highlights:**
- ✅ 87% prediction accuracy (R² Score)
- ✅ Uses only 5 simple features
- ✅ Production-ready saved model
- ✅ Easy-to-use prediction system

---

## ✨ Features

- **Data Exploration:** Comprehensive EDA with visualizations
- **Multiple Models:** Comparison between Linear Regression and Random Forest
- **Feature Importance:** Analysis of which features matter most
- **Saved Model:** Pre-trained model ready for deployment
- **Prediction System:** Input house details and get instant price estimates

---

## 📂 Dataset

**Source:** King County House Sales Dataset  
**Records:** 21,613 house sales  
**Features Used:**
- `bedrooms` - Number of bedrooms
- `bathrooms` - Number of bathrooms
- `sqft_living` - Living area in square feet
- `sqft_lot` - Lot size in square feet
- `floors` - Number of floors

**Target Variable:** `price` (Sale price in USD)

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
```bash
   git clone https://github.com/shubhamjais04/Housewise.git
   cd Housewise
```

2. **Install required libraries:**
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. **Run Jupyter Notebook:**
```bash
   jupyter notebook Housewise.ipynb
```

---

## 💻 Usage

### Running the Complete Analysis:

1. Open the Jupyter notebook
2. Run all cells sequentially (Cell → Run All)
3. Explore the visualizations and model results

### Making Predictions:

Navigate to **Cell 39, 40, or 41** and modify the input values:
```python
# Example prediction
bedrooms = 3
bathrooms = 2
sqft_living = 2000
sqft_lot = 5000
floors = 2

# Run the cell to get predicted price
# Output: Predicted Price: $425,000
```

### Using the Saved Model:
```python
import pickle
import pandas as pd

# Load model and scaler
model = pickle.load(open('models/house_price_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Prepare input
house_data = [[3, 2, 2000, 5000, 2]]  # [bed, bath, sqft_living, sqft_lot, floors]
house_df = pd.DataFrame(house_data, columns=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors'])

# Scale and predict
house_scaled = scaler.transform(house_df)
predicted_price = model.predict(house_scaled)[0]

print(f"Predicted Price: ${predicted_price:,.2f}")
```

---

## 📈 Model Performance

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Linear Regression | 0.65 | $145,000 | $95,000 |
| **Random Forest** | **0.87** | **$85,000** | **$55,000** |

**Winner:** Random Forest Regressor 🏆

**What this means:**
- Model explains **87% of price variation**
- Average prediction error: **$85,000**
- Typical error: **19%** of house price

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical computations
- **matplotlib** - Data visualization
- **seaborn** - Statistical plots
- **scikit-learn** - Machine learning models
- **Jupyter Notebook** - Interactive development

---

## 📁 Project Structure
```
Real-Estate-Price-Prediction/
│
├── Real_Estate_Price_Prediction.ipynb    # Main notebook
├── kc_house_data.csv                      # Dataset
├── models/
│   ├── house_price_model.pkl              # Trained model
│   ├── scaler.pkl                         # Feature scaler
│   └── model_info.pkl                     # Model metadata
│
├── README.md                              # Project documentation
└── LICENSE                                # MIT License
```

---

## 🎯 Results

### Key Findings:

1. **sqft_living (living area) is the most important feature** - 3x more important than others
2. **Random Forest outperforms Linear Regression by 30%**
3. **5 features are sufficient** for good predictions
4. **Model generalizes well** - similar performance on train and test data

### Sample Predictions:

| House Type | Bedrooms | Bathrooms | Living Area | Predicted Price |
|------------|----------|-----------|-------------|-----------------|
| Starter Home | 2 | 1 | 1,200 sqft | $285,000 |
| Average Home | 3 | 2 | 2,000 sqft | $425,000 |
| Luxury Home | 5 | 3.5 | 3,500 sqft | $685,000 |

### Visualizations:

The project includes:
- Price distribution analysis
- Correlation heatmaps
- Feature vs Price scatter plots
- Actual vs Predicted comparisons
- Feature importance charts

---

## 🚀 Future Improvements

- [ ] Add more features (location, condition, grade, view)
- [ ] Implement XGBoost and LightGBM models
- [ ] Create a web application using Streamlit
- [ ] Add confidence intervals for predictions
- [ ] Include time series analysis for price trends
- [ ] Deploy as REST API
- [ ] Add unit tests

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Your Name**
- Email: shubhjais.in@gmailcom
- LinkedIn: [Shubham Jaiswal](https://linkedin.com/in/shubhamjaiswal2004)
- GitHub: [@shubhamjais04](https://github.com/shubhamjais04)

---

## 🙏 Acknowledgments

- King County House Sales Dataset
- Scikit-learn documentation
- Python data science community

---

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐!

---

**Project Status:** Complete ✅  
**Last Updated:** February 2026  
**Version:** 1.0.0

---

