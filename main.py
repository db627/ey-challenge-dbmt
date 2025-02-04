import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

file_path = "data/trainingdata.csv"
df = pd.read_csv(file_path)

# Display basic info
print(df.head())
print(df.info())

df = df.dropna()

# Selecting features (latitude, longitude) and target (UHI Index)
features = ["Latitude", "Longitude"] 
target = "UHI Index"

X = df[features]
y = df[target]

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

feature_importances = model.feature_importances_
for feature, importance in zip(features, feature_importances):
    print(f"{feature}: {importance:.4f}")

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual UHI Index")
plt.ylabel("Predicted UHI Index")
plt.title("Actual vs Predicted UHI Index")
plt.show()
