"""
02_feature_engineering.py
- Loads features.csv and target.csv
- Cleans & engineers features
- Aligns with target
- Saves processed_features.csv and NumPy arrays (X_seq.npy, y_seq.npy)
"""

import os
import pandas as pd
import numpy as np

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = r"C:\Users\Riya\IIT_EDA_Internship\Data\processed_features"
RAW_FEATURES = r"C:\Users\Riya\IIT_EDA_Internship\Data\raw\features.csv"
RAW_TARGET = r"C:\Users\Riya\IIT_EDA_Internship\Data\raw\target.csv"

os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# Load features and target
# ----------------------------
features = pd.read_csv(RAW_FEATURES)
target = pd.read_csv(RAW_TARGET)

print("✅ Loaded features:", features.shape)
print("✅ Loaded target:", target.shape)

# ----------------------------
# Cleaning
# ----------------------------
features = features.drop_duplicates().fillna(method="ffill").fillna(0)
target = target.fillna(method="ffill").fillna(0)

# ----------------------------
# Basic Feature Engineering (examples)
# ----------------------------
if "value" in features.columns:
    features["value_roll_mean3"] = features["value"].rolling(window=3, min_periods=1).mean()
    features["value_diff"] = features["value"].diff().fillna(0)

print("✨ Features engineered:", features.shape)

# ----------------------------
# Align features and target
# ----------------------------
# Ensure same length
min_len = min(len(features), len(target))
features = features.iloc[:min_len].reset_index(drop=True)
target = target.iloc[:min_len].reset_index(drop=True)

# Combine into one DataFrame
processed = features.copy()
processed["target"] = target.values.ravel()

# ----------------------------
# Save processed features
# ----------------------------
csv_path = os.path.join(DATA_DIR, "processed_features.csv")
processed.to_csv(csv_path, index=False)
print("✅ Saved:", csv_path)

# ----------------------------
# Save NumPy arrays (for LSTM / ML models)
# ----------------------------
X = features.values.astype(np.float32)
y = target.values.astype(np.float32)

# Example: reshape X into [samples, timesteps, features]
# Here we assume each row = 1 timestep
# If you want sequences of length 10, adjust below
SEQ_LEN = 10
X_seq, y_seq = [], []
for i in range(len(X) - SEQ_LEN):
    X_seq.append(X[i:i+SEQ_LEN])
    y_seq.append(y[i+SEQ_LEN])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

np.save(os.path.join(DATA_DIR, "X_seq.npy"), X_seq)
np.save(os.path.join(DATA_DIR, "y_seq.npy"), y_seq)

print("✅ Saved X_seq.npy:", X_seq.shape)
print("✅ Saved y_seq.npy:", y_seq.shape)
