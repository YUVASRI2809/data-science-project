import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("Iris.csv")  # Update the filename if needed

# Display basic info
print(df.info())
print(df.head())

# Drop 'Id' column if it exists
df.drop(columns=['Id'], errors='ignore', inplace=True)

# Encode categorical target variable
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Splitting dataset into features and target
X = df.drop(columns=['Species'])  # Features
y = df['Species']  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model - Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict & Evaluate
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Saving predictions for submission
submission = pd.DataFrame({'Actual': le.inverse_transform(y_test), 'Predicted': le.inverse_transform(y_pred)})
submission.to_csv('iris_classification_predictions.csv', index=False)
print("Submission file saved as iris_classification_predictions.csv")
