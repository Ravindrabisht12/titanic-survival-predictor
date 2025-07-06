# 🛳️ Titanic Survival Predictor

A Machine Learning project that predicts whether a passenger on the Titanic would have survived, based on their details like age, gender, class, fare, etc.

Built using:
- **scikit-learn** for model training
- **Streamlit** for the interactive web app
- **pandas**, **joblib**, **matplotlib** for data handling and visualization

---

## 🚀 Features

- Logistic Regression model trained on Titanic dataset
- Real-time prediction via a web UI
- Inputs: age, sex, passenger class, fare, family size, port of embarkation
- Shows prediction + survival probability with progress bar

---

## 🧠 Dataset

From [Kaggle Titanic Challenge](https://www.kaggle.com/c/titanic/data)  
Uses `train.csv` for training.

---

## Install dependencies
```pip install -r requirements.txt```
If python3 is required on your system:
```python3 -m pip install -r requirements.txt```


## Run the Streamlit App
```streamlit run app/app.py```