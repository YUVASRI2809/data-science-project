import sys

# Ensure required libraries are installed
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print("Please install the missing modules using: pip install pandas numpy matplotlib seaborn scikit-learn")
    sys.exit(1)

# Load dataset
df = pd.read_csv("imdb_top_1000.csv")  # Update the filename if needed

# Display basic info
print(df.info())
print(df.head())

# Handling missing values
df.fillna({'Meta_score': df['Meta_score'].median(), 'Gross': df['Gross'].median()}, inplace=True)

# Feature Selection: Selecting relevant columns
selected_features = ['Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4', 'Runtime', 'IMDB_Rating']
df = df[selected_features]

# Encoding categorical variables
label_encoders = {}
for col in ['Genre', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Splitting data
X = df.drop(columns=['IMDB_Rating'])  # Features
y = df['IMDB_Rating']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model - Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict & Evaluate
y_pred_lr = lr_model.predict(X_test)
print("Linear Regression R2 Score:", r2_score(y_test, y_pred_lr))
print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lr)))

# Train Model - Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict & Evaluate
y_pred_rf = rf_model.predict(X_test)
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))

# Saving predictions for submission
submission = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
submission.to_csv('movie_rating_predictions.csv', index=False)
print("Submission file saved as movie_rating_predictions.csv")
