import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
#hello
def main():
    # ---------------------------
    # 1. Load and Prepare Training Data
    # ---------------------------
    training_data = pd.read_csv("data/trainingdata.csv")
    
    # Remove latitude and longitude columns if present (challenge prohibits using them)
    for col in ['Latitude', 'Longitude']:
        if col in training_data.columns:
            training_data.drop(col, axis=1, inplace=True)
    
    # Ensure we have a Date column (used to merge with weather data)
    if 'Date' not in training_data.columns:
        print("Error: 'Date' column not found in training data!")
        return
    training_data['Date'] = pd.to_datetime(training_data['Date'])
    
    # Ensure we have a Region column to later merge with building data
    if 'Region' not in training_data.columns:
        print("Error: 'Region' column not found in training data!")
        return
    
    # ---------------------------
    # 2. Incorporate Weather Data
    # ---------------------------
    weather_data = pd.read_excel("data/NY_Mesonet_Weather.xlsx")
    if 'Date' not in weather_data.columns:
        print("Error: 'Date' column not found in weather data!")
        return
    weather_data['Date'] = pd.to_datetime(weather_data['Date'])
    
    # Merge training data with weather data on Date
    merged_data = pd.merge(training_data, weather_data, on='Date', how='left')
    
    # ---------------------------
    # 3. Process Building Footprint Data
    # ---------------------------
    # Read building footprints using geopandas (requires a proper driver for KML)
    building_gdf = gpd.read_file("data/Building_Footprint.kml", driver='KML')
    
    # Convert to a projected CRS for accurate area computation
    building_gdf = building_gdf.to_crs(epsg=3857)
    # Compute area for each building footprint
    building_gdf['building_area'] = building_gdf.geometry.area
    
    # Assume building_gdf has a 'Region' field (e.g., "Manhattan", "Bronx")
    if 'Region' not in building_gdf.columns:
        print("Error: 'Region' column not found in building footprint data!")
        return

    # Calculate building statistics per region
    building_stats = building_gdf.groupby('Region').agg(
        building_count=('Region', 'count'),
        total_building_area=('building_area', 'sum')
    ).reset_index()
    
    # ---------------------------
    # 4. Merge Building Stats with Main Data
    # ---------------------------
    merged_data = pd.merge(merged_data, building_stats, on='Region', how='left')
    
    # Drop any rows with missing values after the merge
    merged_data = merged_data.dropna()
    
    # ---------------------------
    # 5. Define Features and Target
    # ---------------------------
    # Our target remains the UHI Index
    target = 'UHI Index'
    
    # Use weather variables and building statistics as features.
    # Adjust these column names based on your actual weather dataset.
    features = ['Temperature', 'Humidity', 'WindSpeed', 'building_count', 'total_building_area']
    
    # Verify all selected features exist in the merged dataframe
    missing_features = [feat for feat in features if feat not in merged_data.columns]
    if missing_features:
        print(f"Error: Missing features in merged data: {missing_features}")
        return
    
    X = merged_data[features]
    y = merged_data[target]
    
    # ---------------------------
    # 6. Split Data and Train the Model
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # ---------------------------
    # 7. Evaluate the Model
    # ---------------------------
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse}")
    print(f"R² Score: {r2}")
    
    # Display feature importances
    feature_importances = model.feature_importances_
    for feat, importance in zip(features, feature_importances):
        print(f"{feat}: {importance:.4f}")
    
    # ---------------------------
    # 8. Plot Actual vs. Predicted UHI Index
    # ---------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual UHI Index")
    plt.ylabel("Predicted UHI Index")
    plt.title("Actual vs. Predicted UHI Index")
    plt.show()

if __name__ == "__main__":
    main()
