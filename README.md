# 💧 WaterPotability AI

### Machine Learning-Based Water Quality Classification System

An interactive Machine Learning application that analyzes water-quality measurements and predicts water potability labels using a Random Forest Classifier.

## Features

- Single water sample prediction
- Bulk CSV upload for unseen data
- Interactive Streamlit dashboard
- Prediction probabilities and downloadable results
- Feature importance visualization
- Cinematic dark-themed interface

## Technologies Used

Python | Pandas | NumPy | Scikit-learn | Streamlit | Plotly | Joblib

## Dataset

**Water Quality Dataset** — 3,276 samples with nine input features and one target variable.

Source: [Kaggle Water Potability Dataset](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

## Machine Learning Models

Two algorithms were evaluated:

- Logistic Regression
- Random Forest Classifier

**Final Random Forest results:**

| Metric | Score |
|---|---|
| Accuracy | 67.07% |
| Non-potable recall | 91% |
| Potable recall | 30% |

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

## Project Workflow

Data Collection → Data Cleaning → Exploratory Data Analysis → Model Training → Evaluation → Streamlit Application

## Limitations

The dataset's provenance is uncertain, and the model has limited predictive performance. This project is intended for educational purposes and is not a substitute for certified laboratory testing or drinking-water safety assessment.
