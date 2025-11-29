import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import joblib

data_path = os.path.join("..", "data", "yield_regression_data.csv")
model_path = os.path.join("..", "models", "rf_regressor.joblib")

os.makedirs(os.path.join("..", "models"), exist_ok=True)

def generate_synthetic_data(num_samples=100):
    data = {
        'Severity_Score': np.random.uniform(0.1, 0.9, num_samples),
        'Crop_Type': np.random.choice(['Tomato', 'Pepper', 'Potato'], num_samples),
        'Yield_Loss_Percentage': (data['Severity_Score'] * 50) + np.random.normal(0, 5, num_samples)
    }
    df = pd.DataFrame(data)
    df['Yield_Loss_Percentage'] = np.clip(df['Yield_Loss_Percentage'], 0, 80)
    return df

try:
    df = pd.read_csv(data_path)
    if 'Severity_Score' not in df.columns or len(df) <= 1:
        raise FileNotFoundError
except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
    print("Generating synthetic data for the Regression model. REPLACE THIS LATER.")
    df = generate_synthetic_data(num_samples=200)

df_processed = pd.get_dummies(df, columns=['Crop_Type'], drop_first=True)

X = df_processed.drop('Yield_Loss_Percentage', axis=1)
y = df_processed['Yield_Loss_Percentage']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Training Random Forest Regressor for Yield Loss Prediction ---")
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

joblib.dump(regressor, model_path)
print(f"Random Forest Regressor saved to {model_path}")

y_pred = regressor.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Regression Performance Metrics (Yield Prediction) ---")
print(f"1. MSE (Mean Squared Error): {mse:.4f}")
print(f"2. RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"3. R-squared (R²): {r2:.4f}")
print(f"4. MAE (Mean Absolute Error): {mae:.4f}")
