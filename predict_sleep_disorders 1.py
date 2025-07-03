# Title: Predicting Sleep Disorders Using Machine Learning on Health and Lifestyle Data
# Authors: Nallavelli Shravan Kumar, Shyam Verma, Pooja Shinde
# Institution: Northwood University
# Date: July 2, 2025

# --------------------------------------------------
# Step 1: Import necessary libraries
# --------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --------------------------------------------------
# Step 2: Load the dataset
# --------------------------------------------------
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")  # Update with the actual path

# --------------------------------------------------
# Step 3: Exploratory Data Analysis (EDA)
# --------------------------------------------------
print("Dataset Head:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())

# --------------------------------------------------
# Step 4: Encode categorical variables
# --------------------------------------------------
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# --------------------------------------------------
# Step 5: Feature Scaling
# --------------------------------------------------
scaler = StandardScaler()
X = df.drop('Sleep Disorder', axis=1)
y = df['Sleep Disorder']
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# Step 6: Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --------------------------------------------------
# Step 7: Train models
# --------------------------------------------------
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

svc = SVC(kernel='linear', probability=True)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)

# --------------------------------------------------
# Step 8: Evaluate models
# --------------------------------------------------
print("\nRandom Forest Results:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nSVM Results:")
print(confusion_matrix(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))
print("Accuracy:", accuracy_score(y_test, y_pred_svc))

# --------------------------------------------------
# Step 9: Plot Feature Importance (Random Forest)
# --------------------------------------------------
importances = rf.feature_importances_
features = X.columns
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()
