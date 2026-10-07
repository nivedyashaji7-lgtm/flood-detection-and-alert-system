# 🌊 Flood Detection & Alert System

A machine learning-based web application that predicts flood probability and provides a flood risk alert based on environmental, infrastructure, and population-related factors.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask
- SQLite
- HTML/CSS

## 🔍 Features

- Flood probability prediction
- Low, Moderate, and High risk classification
- XGBoost machine learning model
- Flask web interface
- SQLite database for storing predictions

## 📊 Model Performance

- **MAE:** 0.001600
- **RMSE:** 0.002207
- **R² Score:** 0.998044

## 📁 Project Structure

```text
flood-detection-alert-system/
├── app.py
├── data_analysis.py
├── feature_selection.py
├── group_feature_test.py
├── train_final_model.py
├── final_model.pkl
├── flood.xlsx
└── templates/
    └── index.html
```

## 🚀 How to Run

```bash
pip install pandas numpy scikit-learn xgboost flask openpyxl joblib
python app.py
```

Then open the local Flask URL in your browser.

## 👩‍💻 Project

**Flood Detection & Alert System**

![Flood Detection System](screenshots/ flood%20system.png)
