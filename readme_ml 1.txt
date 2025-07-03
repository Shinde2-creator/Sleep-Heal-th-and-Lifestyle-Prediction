# Predicting Sleep Disorders Using Machine Learning on Health and Lifestyle Data

## Authors
Nallavelli Shravan Kumar, Shyam Verma, Pooja Shinde  
Northwood University  
Professor: Dr. Itauama  
Date: July 2, 2025

---

## Overview
This project focuses on using machine learning techniques to predict the presence of sleep disorders based on individuals' health and lifestyle data. The goal is to build classification models that can accurately identify whether a person is likely to suffer from sleep disorders like insomnia, sleep apnea, etc.

---

## Dataset
**Filename**: `Sleep_health_and_lifestyle_dataset.csv`  
The dataset includes features such as age, gender, BMI category, physical activity, stress levels, and sleep duration. The target variable is the type of sleep disorder.

---

## Steps Performed
1. **Data Cleaning & Preprocessing**
   - Encoding categorical variables
   - Handling missing values (if any)
   - Feature scaling using StandardScaler

2. **Model Building**
   - Random Forest Classifier
   - Support Vector Machine (Linear Kernel)

3. **Model Evaluation**
   - Accuracy, Confusion Matrix, Classification Report
   - Feature Importance Plot for Random Forest

4. **Visualization**
   - Feature importance using Seaborn bar plots

---

## Requirements
Install the dependencies using:
```bash
pip install -r requirements.txt
```

Main libraries used:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Results
- **Random Forest Accuracy:** [displayed in output]
- **SVM Accuracy:** [displayed in output]
- Feature Importance: Visualized and saved from Random Forest

---

## How to Run
1. Place the dataset in the same folder as the script.
2. Run the script:
```bash
python predict_sleep_disorders.py
```
3. Review the printed output and generated plots.

---

## Notes
- This script is designed for academic purposes as part of a group assignment at Northwood University.
- Only one member submitted the code file; the other submitted the GitHub repository link.

---

## Contact
For questions, contact any of the contributors or Dr. Itauama at Northwood University.
