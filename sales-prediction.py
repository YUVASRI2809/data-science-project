import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("Advertising.csv")  # Update the filename if needed

# Display basic info
print(df.info())
print(df.head())

# Drop unnecessary column if it exists
df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

# Feature Selection: Selecting relevant columns
X = df[['TV', 'Radio', 'Newspaper']]  # Independent variables
y = df['Sales']  # Dependent variable

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model - Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predicting sales
y_pred = lr_model.predict(X_test)

# Evaluating the model
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Visualizing Predictions
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Saving predictions for submission
submission = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})
submission.to_csv('sales_predictions.csv', index=False)
print("Submission file saved as sales_predictions.csv")
