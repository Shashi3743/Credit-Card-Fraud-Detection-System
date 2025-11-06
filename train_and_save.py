import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
data = pd.read_csv("card_transdata.csv")  # Save this CSV locally

X = data.drop("fraud", axis=1)
y = data["fraud"]

# Preprocess
continuous_features = ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']
scaler = StandardScaler()
X[continuous_features] = scaler.fit_transform(X[continuous_features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
