"""
04_modeling_baseline_ml.py
- Loads X_seq.npy / y_seq.npy
- Trains scikit-learn regressors (RandomForest, GradientBoosting, LinearRegression)
- Reports MAE, RMSE, R2
- Saves best model as joblib file
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
DATA_DIR = r"C:\Users\Riya\IIT_EDA_Internship\Data\processed_features"
MODEL_DIR = r"C:\Users\Riya\IIT_EDA_Internship\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
X = np.load(os.path.join(DATA_DIR, "X_seq.npy"))
y = np.load(os.path.join(DATA_DIR, "y_seq.npy")).ravel()
print("Loaded X, y:", X.shape, y.shape)

# Flatten sequences for ML models (scikit-learn expects 2D input)
X = X.reshape(X.shape[0], -1)

# Train/val split (80/20, time-ordered)
N = len(X)
split = int(0.8 * N)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# Define candidate models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "LinearRegression": LinearRegression()
}

best_mae = float("inf")
best_model_name = None
best_model = None

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    r2 = r2_score(y_val, y_pred)

    print(f"\n{name} Performance:")
    print(f"MAE: {mae:.5f} | RMSE: {rmse:.5f} | R2: {r2:.5f}")

    if mae < best_mae:
        best_mae = mae
        best_model_name = name
        best_model = model

# Save best model
save_path = os.path.join(MODEL_DIR, f"{best_model_name}_best.pkl")
joblib.dump(best_model, save_path)
print(f"\n✅ Best Model: {best_model_name} (MAE={best_mae:.5f}) saved at {save_path}")
