# optimized-weather-classification-xgboost-GeneticA
Optimized Weather Classification using XGBoost and Genetic Algorithm

# Optimized Weather Classification using XGBoost & Genetic Algorithm

## 📌 Project Overview
This project presents an optimized weather type classification system using XGBoost combined with Genetic Algorithm–based hyperparameter optimization.

## 🧠 Technologies Used
- Python
- XGBoost
- Genetic Algorithm
- Scikit-learn
- Streamlit
- Pandas, NumPy, Matplotlib

## 🌦️ Weather Classes
- Cloudy
- Rainy
- Snowy
- Sunny

## 📊 Performance Comparison
| Model | Training Accuracy | Validation Accuracy |
|------|------------------|--------------------|
| Baseline XGBoost | 90.89% | 90.23% |
| GA-Optimized XGBoost | 96.55% | 92.12% |

## 🖥️ Streamlit GUI
The project includes an interactive Streamlit web application that allows users to input weather features and receive real-time predictions.

## 🚀 How to Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app/GUI.py
