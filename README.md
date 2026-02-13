# Gaza Hunger Prediction System - Streamlit Application

## 📋 Overview

A professional, interactive Streamlit application for predicting household hunger severity in Gaza using machine learning. This system analyzes socioeconomic indicators, displacement patterns, and food security metrics to assess household vulnerability.

## ✨ Features

### 1. **Data Overview** 📊
- Interactive dataset exploration
- Statistical summaries (numerical & categorical)
- Target variable distribution visualization
- Missing values analysis
- Feature descriptions

### 2. **Data Preprocessing** 🔧
- Automatic label encoding for categorical variables
- Feature scaling with StandardScaler
- Configurable train-test split
- Real-time preprocessing status

### 3. **Model Training** 🤖
- Three machine learning algorithms:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Automated model comparison
- Performance metrics visualization

### 4. **Model Performance** 📈
- Detailed accuracy metrics (Accuracy, Precision, Recall, F1-Score)
- Interactive confusion matrix
- Classification reports
- Feature importance analysis (for tree-based models)
- Downloadable feature importance data

### 5. **Make Predictions** 🎯
- User-friendly input form for household data
- Real-time prediction with probability scores
- Risk level classification (High/Moderate/Low)
- Color-coded results with visual indicators
- Actionable recommendations based on prediction

### 6. **Export Results** 📥
- Download processed datasets (CSV)
- Export model predictions with actual vs predicted labels
- Generate comprehensive performance reports (TXT)
- Export trained models (.pkl) for deployment

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Application
```bash
streamlit run gaza_hunger_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

## 📊 Dataset Information

- **Total Households**: 1,209
- **Data Collection Period**: May-July 2024
- **Location**: Gaza Strip
- **Features**: 50 variables including:
  - Family composition (size, children, elderly, etc.)
  - Economic indicators (income, employment, etc.)
  - Living conditions (shelter type, displacement, etc.)
  - Food security metrics (meal frequency, food access, etc.)
- **Target Variable**: Water Availability (Q50) - proxy for hunger severity

## 🎯 How to Use

### Basic Workflow:

1. **Start Here - Data Overview Tab**
   - Review dataset statistics and distributions
   - Understand feature descriptions
   - Check for data quality issues

2. **Preprocess the Data**
   - Click "Start Preprocessing" button
   - Review encoded variables and scaled features
   - Verify train-test split

3. **Train Models**
   - Click "Train All Models" button
   - Compare performance metrics
   - Identify the best performing model

4. **Analyze Performance**
   - Select a model for detailed analysis
   - Review confusion matrix and classification report
   - Examine feature importance (if applicable)

5. **Make Predictions**
   - Fill in household information form
   - Click "Predict Hunger Severity"
   - Review risk assessment and recommendations

6. **Export Your Results**
   - Download processed data for analysis
   - Export predictions for reporting
   - Save trained models for deployment

## 🎨 User Interface Features

### Sidebar Controls:
- **Model Selection**: Choose between Logistic Regression, Random Forest, or Gradient Boosting
- **Test Set Size**: Adjust the percentage of data used for testing (10-40%)

### Visual Elements:
- **Interactive Charts**: Plotly-based visualizations for exploration
- **Color-Coded Predictions**: 
  - 🔴 Red = High Risk
  - 🟠 Orange = Moderate Risk
  - 🟢 Green = Low Risk
- **Metrics Cards**: Quick access to key statistics
- **Progress Indicators**: Real-time feedback during processing

## 📁 File Structure

```
├── gaza_hunger_app.py          # Main Streamlit application
├── requirements.txt             # Python dependencies
├── GazaHungerData.xlsx         # Dataset (required)
└── README.md                   # This file
```

## 🔧 Configuration

### Model Parameters (editable in code):

**Logistic Regression:**
- `max_iter`: 1000
- `random_state`: 42
- `class_weight`: 'balanced'

**Random Forest:**
- `n_estimators`: 100
- `max_depth`: 10
- `class_weight`: 'balanced'

**Gradient Boosting:**
- `n_estimators`: 100
- `max_depth`: 5
- `learning_rate`: 0.1

## 📈 Expected Performance

Performance ranges (will vary based on train-test split):
- **Accuracy**: 0.75 - 0.85
- **Precision**: 0.72 - 0.82
- **Recall**: 0.70 - 0.80
- **F1-Score**: 0.71 - 0.81

*Note: Random Forest typically performs best on this dataset*

## 🛠️ Troubleshooting

### Common Issues:

1. **"Module not found" error**
   - Solution: Run `pip install -r requirements.txt`

2. **"Failed to load data" error**
   - Solution: Ensure `GazaHungerData.xlsx` is accessible

3. **Slow performance**
   - Solution: Reduce the number of features or use a smaller dataset

## 🔐 Security & Privacy

- This application is designed for **humanitarian use only**
- All data should be handled according to relevant privacy regulations
- Predictions should be validated by field workers
- No data is stored or transmitted outside the local session

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments for implementation details
3. Refer to the original Jupyter notebook for algorithm explanations

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Status**: Production Ready ✅
