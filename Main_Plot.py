import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from shapely.geometry import Point
import geopandas as gpd
import rioxarray
import xarray as xr
import rasterio

# For modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Directories
DATA_DIR = "data"
FIG_DIR = "figures"
for d in [DATA_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

###############################################################################
# 1) HELPER FUNCTIONS FOR DATA LOADING & FEATURE ENGINEERING
###############################################################################

def load_uhi_data(uhi_csv="trainingdata.csv"):
    """
    Load the main UHI training data. 
    Returns a DataFrame with columns: Longitude, Latitude, datetime (as string), UHI Index
    """
    df = pd.read_csv(uhi_csv)
    # Adjust parsing if needed, e.g. dayfirst=True or exact format
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M', errors='coerce')
    return df

def load_validation_data(validation_csv):
    """
    Load validation data from a CSV file.
    Expected format similar to training data.
    """
    print(f"Loading validation data from {validation_csv}...")
    try:
        df_val = pd.read_csv(validation_csv)
        df_val['datetime'] = pd.to_datetime(df_val['datetime'], format='%d-%m-%Y %H:%M', errors='coerce')
        return df_val
    except Exception as e:
        print(f"Error loading validation data: {e}")
        return None

def load_weather_data(weather_xlsx="NY_Mesonet_Weather.xlsx"):
    """
    Loads Bronx & Manhattan weather data from Excel, returns two DataFrames.
    """
    bronx = pd.read_excel(weather_xlsx, sheet_name='Bronx')
    manhattan = pd.read_excel(weather_xlsx, sheet_name='Manhattan')

    # Parse times
    bronx['datetime'] = pd.to_datetime(bronx['Date / Time'], errors='coerce')
    manhattan['datetime'] = pd.to_datetime(manhattan['Date / Time'], errors='coerce')

    # Rename columns to unify
    bronx.rename(columns={
        'Air Temp at Surface [degC]': 'air_temp_bronx',
        'Relative Humidity [percent]': 'humidity_bronx',
        'Avg Wind Speed [m/s]': 'wind_speed_bronx',
        'Wind Direction [degrees]': 'wind_dir_bronx',
        'Solar Flux [W/m^2]': 'solar_flux_bronx'
    }, inplace=True)

    manhattan.rename(columns={
        'Air Temp at Surface [degC]': 'air_temp_manhattan',
        'Relative Humidity [percent]': 'humidity_manhattan',
        'Avg Wind Speed [m/s]': 'wind_speed_manhattan',
        'Wind Direction [degrees]': 'wind_dir_manhattan',
        'Solar Flux [W/m^2]': 'solar_flux_manhattan'
    }, inplace=True)

    return bronx, manhattan

def nearest_time_merge(uhi_df, weather_df, time_col_uhi='datetime', time_col_wx='datetime',
                       max_time_diff='10min', suffix=''):
    """
    Perform a nearest-time merge between UHI data & single weather DataFrame.
    For each row in uhi_df, find the weather row whose time is within ± max_time_diff 
    and is closest in time. Returns a merged DataFrame with the weather columns.
    """
    # Sort by time
    uhi_df = uhi_df.sort_values(by=time_col_uhi).copy()
    weather_df = weather_df.sort_values(by=time_col_wx).copy()

    merged_rows = []
    max_diff = pd.to_timedelta(max_time_diff)

    j = 0
    w_times = weather_df[time_col_wx].values

    for i in range(len(uhi_df)):
        u_time = uhi_df.iloc[i][time_col_uhi]
        # Advance pointer in weather_df so w_times[j] is the closest to u_time
        while j < len(w_times)-1 and abs((w_times[j+1] - u_time)) < abs((w_times[j] - u_time)):
            j += 1
        # Check difference
        time_diff = abs(weather_df.iloc[j][time_col_wx] - u_time)
        if time_diff <= max_diff:
            row_merged = {**uhi_df.iloc[i].to_dict(), **weather_df.iloc[j].to_dict()}
        else:
            # out of tolerance => fill weather with NaN
            row_merged = uhi_df.iloc[i].to_dict()
            for c in weather_df.columns:
                if c == time_col_wx:
                    continue
                row_merged[c] = np.nan
        merged_rows.append(row_merged)

    merged_df = pd.DataFrame(merged_rows)

    if suffix:
        w_cols = [c for c in weather_df.columns if c != time_col_wx]
        rename_map = {c: c + suffix for c in w_cols}
        merged_df.rename(columns=rename_map, inplace=True)

    return merged_df

def combine_bronx_manhattan_weather(uhi_df, bronx_df, manhattan_df):
    """
    1. Merge UHI with Bronx data via nearest_time_merge
    2. Merge that with Manhattan data
    3. Compute average columns
    """
    merged_bronx = nearest_time_merge(uhi_df, bronx_df, max_time_diff='10min')
    final_merged = nearest_time_merge(merged_bronx, manhattan_df, max_time_diff='10min', suffix='_man')

    # Compute averages if they exist
    if 'air_temp_bronx' in final_merged.columns and 'air_temp_manhattan_man' in final_merged.columns:
        final_merged['avg_air_temp'] = final_merged[['air_temp_bronx','air_temp_manhattan_man']].mean(axis=1)
    if 'humidity_bronx' in final_merged.columns and 'humidity_manhattan_man' in final_merged.columns:
        final_merged['avg_humidity'] = final_merged[['humidity_bronx','humidity_manhattan_man']].mean(axis=1)
    if 'wind_speed_bronx' in final_merged.columns and 'wind_speed_manhattan_man' in final_merged.columns:
        final_merged['avg_wind_speed'] = final_merged[['wind_speed_bronx','wind_speed_manhattan_man']].mean(axis=1)
    if 'solar_flux_bronx' in final_merged.columns and 'solar_flux_manhattan_man' in final_merged.columns:
        final_merged['avg_solar_flux'] = final_merged[['solar_flux_bronx','solar_flux_manhattan_man']].mean(axis=1)

    return final_merged

def multi_buffer_building_coverage(df, building_kml="Building_Footprint.kml", buffers=[50,100,200]):
    """
    For each (lat/lon) row, compute coverage fraction of building footprints
    within multiple buffer radii (e.g. 50m, 100m, 200m).
    """
    try:
        gdf_buildings = gpd.read_file(building_kml).to_crs(epsg=2263)

        gdf_points = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
            crs="EPSG:4326"
        ).to_crs(epsg=2263)

        for buf_m in buffers:
            colname = f"bldg_cov_{buf_m}m"
            coverage_list = []
            gdf_points["buffer_geom"] = gdf_points.geometry.buffer(buf_m)
            for _, row in gdf_points.iterrows():
                clipped = gdf_buildings.clip(row["buffer_geom"])
                if clipped.empty:
                    coverage_list.append(0.0)
                else:
                    coverage_list.append(clipped.area.sum() / row["buffer_geom"].area)
            gdf_points[colname] = coverage_list

        gdf_points.drop(columns=["buffer_geom","geometry"], inplace=True)
        return pd.DataFrame(gdf_points)
    except Exception as e:
        print(f"Error computing building coverage: {e}")
        return df

def add_spatial_features(df):
    """
    Add lat2, lon2, lat_lon features for capturing spatial curvature
    """
    df["lat2"] = df["Latitude"]**2
    df["lon2"] = df["Longitude"]**2
    df["lat_lon"] = df["Latitude"] * df["Longitude"]
    return df


###############################################################################
# 2) MAIN MODEL FUNCTION
###############################################################################

def run_uhi_analysis(return_features=False):
    """
    1) Load & merge data
    2) Feature engineering
    3) Train RandomForest with hyperparam search
    4) Return the final DataFrame w/ predictions
    
    If return_features is True, also return the feature list and scaler
    """
    print("Loading UHI data...")
    df_uhi = load_uhi_data("trainingdata.csv")
    print(f"Initial UHI shape: {df_uhi.shape}")

    # Basic time features
    df_uhi["hour"] = df_uhi["datetime"].dt.hour
    df_uhi["minute"] = df_uhi["datetime"].dt.minute
    df_uhi["day_of_year"] = df_uhi["datetime"].dt.dayofyear

    df_uhi["minutes_since_noon"] = (df_uhi["hour"] - 12)*60 + df_uhi["minute"]
    total_minutes_day = 24*60
    df_uhi["time_of_day"] = np.sin(2*np.pi*(df_uhi["hour"]*60 + df_uhi["minute"])/total_minutes_day)

    # Merge with weather
    try:
        bronx_df, manhattan_df = load_weather_data("NY_Mesonet_Weather.xlsx")
        df_merged = combine_bronx_manhattan_weather(df_uhi, bronx_df, manhattan_df)
    except Exception as e:
        print(f"Weather data load/merge error: {e}")
        df_merged = df_uhi.copy()

    # Distances to stations if available
    if "air_temp_bronx" in df_merged.columns and "air_temp_manhattan_man" in df_merged.columns:
        def haversine(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371
            return c*r

        bronx_lat, bronx_lon = 40.8679, -73.8806
        man_lat,  man_lon   = 40.7893, -73.9631

        df_merged["dist_bronx"] = df_merged.apply(
            lambda r: haversine(r["Latitude"], r["Longitude"], bronx_lat, bronx_lon), axis=1
        )
        df_merged["dist_manh"] = df_merged.apply(
            lambda r: haversine(r["Latitude"], r["Longitude"], man_lat, man_lon), axis=1
        )

        # Weighted temperature
        df_merged["temp_bronx"] = df_merged["air_temp_bronx"]
        df_merged["temp_manh"]  = df_merged["air_temp_manhattan_man"]
        df_merged["weighted_temp"] = (
            df_merged["temp_bronx"]/(df_merged["dist_bronx"]+0.001) + 
            df_merged["temp_manh"]/(df_merged["dist_manh"]+0.001)
        ) / (
            1/(df_merged["dist_bronx"]+0.001) + 1/(df_merged["dist_manh"]+0.001)
        )

    # Building coverage
    df_merged = multi_buffer_building_coverage(df_merged, "Building_Footprint.kml", [50,100,200])

    # Add lat/lon expansions
    df_merged = add_spatial_features(df_merged)

    # Target
    target_col = "UHI Index"
    df_merged.dropna(subset=[target_col], inplace=True)

    # Candidate Features
    candidate_feats = [
        "Latitude","Longitude","lat2","lon2","lat_lon",
        "hour","minute","day_of_year","minutes_since_noon","time_of_day",
        "air_temp_bronx","humidity_bronx","wind_speed_bronx","wind_dir_bronx","solar_flux_bronx",
        "air_temp_manhattan_man","humidity_manhattan_man","wind_speed_manhattan_man","wind_dir_manhattan_man","solar_flux_manhattan_man",
        "avg_air_temp","avg_humidity","avg_wind_speed","avg_solar_flux",
        "dist_bronx","dist_manh","weighted_temp",
        "bldg_cov_50m","bldg_cov_100m","bldg_cov_200m"
    ]
    # Keep only what's actually present
    features = [f for f in candidate_feats if f in df_merged.columns]

    # Fill missing
    for c in features:
        df_merged[c].fillna(df_merged[c].mean(), inplace=True)

    X = df_merged[features].copy()
    y = df_merged[target_col].copy()

    # Example polynomial expansions
    poly_cols = ["time_of_day","bldg_cov_100m"]
    poly_cols = [c for c in poly_cols if c in X.columns]

    if poly_cols:
        pf = PolynomialFeatures(degree=2, include_bias=False)
        subX = X[poly_cols].values
        subX_poly = pf.fit_transform(subX)
        poly_names = pf.get_feature_names_out(poly_cols)
        df_poly = pd.DataFrame(subX_poly, columns=poly_names, index=X.index)
        X = pd.concat([X.drop(columns=poly_cols), df_poly], axis=1)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("\nTuning Random Forest with more exhaustive search...")
    rf = RandomForestRegressor(random_state=42)
    param_dist = {
        "n_estimators":      [200, 300, 400, 500, 600],
        "max_depth":         [5,10,20,None],
        "min_samples_split": [2,4,6,10],
        "min_samples_leaf":  [1,2,4],
        "max_features":      ["auto","sqrt","log2"]
    }
    rf_search = RandomizedSearchCV(
        rf, param_dist, n_iter=40, cv=5, scoring='r2', 
        n_jobs=-1, verbose=1, random_state=42
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    y_pred = best_rf.predict(X_test)

    r2_val = r2_score(y_test, y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"Best RF params: {rf_search.best_params_}")
    print(f"RF R² on test: {r2_val:.4f}, RMSE: {rmse_val:.4f}")

    # Create output DF with predictions
    test_idx = y_test.index
    df_out = df_merged.loc[test_idx, ["Latitude","Longitude",target_col]].copy()
    df_out["Predicted_UHI"] = y_pred
    df_out["Residual"] = df_out[target_col] - df_out["Predicted_UHI"]

    # Export predictions with detailed information
    export_detailed_predictions(df_out, best_rf, r2_val, rmse_val, X.columns)

    if return_features:
        return df_out, r2_val, best_rf, X.columns, scaler
    else:
        return df_out, r2_val, best_rf

def export_detailed_predictions(df_pred, model, r2_val, rmse_val, features):
    """
    Export detailed prediction results to Excel with multiple sheets:
    1. Predictions - The prediction results
    2. Model Info - Performance metrics and parameters
    3. Feature Importance - Importance of each feature
    """
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_xlsx = os.path.join(DATA_DIR, f"UHI_Predictions_{timestamp}.xlsx")
    
    # Create Excel writer
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        # Sheet 1: Predictions
        df_pred.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Sheet 2: Model Information
        model_info = {
            'Metric': ['R-squared', 'RMSE', 'Number of features', 'Model type'],
            'Value': [r2_val, rmse_val, len(features), type(model).__name__]
        }
        
        # Add model parameters
        for param, value in model.get_params().items():
            model_info['Metric'].append(f"Parameter: {param}")
            model_info['Value'].append(str(value))
        
        pd.DataFrame(model_info).to_excel(writer, sheet_name='Model Info', index=False)
        
        # Sheet 3: Feature Importance
        if hasattr(model, 'feature_importances_'):
            feat_imp = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            feat_imp.to_excel(writer, sheet_name='Feature Importance', index=False)
    
    print(f"\nDetailed predictions exported to {out_xlsx}")
    return out_xlsx

def evaluate_on_validation(model, validation_df, features, scaler=None):
    """
    Apply the trained model to validation data and evaluate performance.
    
    Parameters:
    - model: The trained model
    - validation_df: DataFrame with validation data
    - features: List of feature names used in training
    - scaler: Optional scaler used in training
    
    Returns:
    - DataFrame with original data and predictions
    - Performance metrics
    """
    # Ensure validation data has the same features as training data
    missing_features = [f for f in features if f not in validation_df.columns]
    if missing_features:
        print(f"Warning: Validation data missing features: {missing_features}")
        for feat in missing_features:
            validation_df[feat] = 0  # Fill with default value
    
    # Extract features and target
    X_val = validation_df[features].copy()
    if "UHI Index" in validation_df.columns:
        y_val = validation_df["UHI Index"].copy()
        has_true_values = True
    else:
        y_val = None
        has_true_values = False
    
    # Scale features if a scaler was provided
    if scaler is not None:
        X_val_scaled = scaler.transform(X_val)
    else:
        X_val_scaled = X_val
    
    # Make predictions
    y_pred = model.predict(X_val_scaled)
    
    # Create output DataFrame
    df_out = validation_df[["Latitude", "Longitude"]].copy()
    df_out["Predicted_UHI"] = y_pred
    
    # Calculate metrics if true values are available
    metrics = {}
    if has_true_values:
        df_out["True_UHI"] = y_val
        df_out["Residual"] = df_out["True_UHI"] - df_out["Predicted_UHI"]
        
        r2_val = r2_score(y_val, y_pred)
        rmse_val = np.sqrt(mean_squared_error(y_val, y_pred))
        mae_val = np.mean(np.abs(df_out["Residual"]))
        
        metrics = {
            "R2": r2_val,
            "RMSE": rmse_val,
            "MAE": mae_val
        }
        
        print(f"Validation metrics: R² = {r2_val:.4f}, RMSE = {rmse_val:.4f}, MAE = {mae_val:.4f}")
    
    return df_out, metrics

###############################################################################
# 3) VISUALIZATION: PLOT PREDICTIONS ON A SATELLITE BASEMAP
###############################################################################

def plot_uhi_basemap(df_pred, lat_col="Latitude", lon_col="Longitude", pred_col="Predicted_UHI"):
    """
    Plots the predicted UHI from df_pred onto a satellite basemap
    (using contextily).
    """
    # 1) Create a colormap from blue->green->yellow->orange->red
    uhi_cmap = LinearSegmentedColormap.from_list("uhi_cmap", 
        ["blue","green","yellow","orange","red"], N=256)
    
    # 2) Convert to GeoDataFrame in EPSG:4326
    gdf = gpd.GeoDataFrame(
        df_pred.copy(),
        geometry=gpd.points_from_xy(df_pred[lon_col], df_pred[lat_col]),
        crs="EPSG:4326"
    )
    # 3) Reproject to Web Mercator (EPSG:3857) for contextily
    gdf_web = gdf.to_crs(epsg=3857)
    
    # 4) Plot
    fig, ax = plt.subplots(figsize=(12,10))
    gdf_web.plot(
        column=pred_col,
        cmap=uhi_cmap,
        alpha=0.8,
        s=30,
        edgecolor='k',
        linewidth=0.4,
        legend=False,
        ax=ax
    )
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    ax.set_axis_off()
    ax.set_title(f"{pred_col} with Satellite Basemap\n(Bronx and Manhattan)", fontsize=16)

    # Add colorbar
    vmin, vmax = gdf_web[pred_col].min(), gdf_web[pred_col].max()
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=uhi_cmap)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(pred_col, fontsize=14)

    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, f"{pred_col.lower().replace(' ', '_')}_basemap.png")
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Map saved to {out_fig}")

###############################################################################
# 4) MAIN
###############################################################################

def main():
    # 1) Run the improved model pipeline
    df_pred, r2_val, model = run_uhi_analysis()
    print(f"\nFinal R²: {r2_val:.3f}")

    # 2) Plot predictions on basemap
    plot_uhi_basemap(df_pred)
    
    # 3) Test on validation data if available
    validation_file = os.path.join(DATA_DIR, "validation_data.csv")
    if os.path.exists(validation_file):
        print("\nTesting model on validation data...")
        
        # Load validation data
        df_val = load_validation_data(validation_file)
        
        if df_val is not None:
            # Prepare validation data (same preprocessing as training data)
            df_val["hour"] = df_val["datetime"].dt.hour
            df_val["minute"] = df_val["datetime"].dt.minute
            df_val["day_of_year"] = df_val["datetime"].dt.dayofyear
            df_val["minutes_since_noon"] = (df_val["hour"] - 12)*60 + df_val["minute"]
            df_val["time_of_day"] = np.sin(2*np.pi*(df_val["hour"]*60 + df_val["minute"])/(24*60))
            
            # Additional preprocessing steps (match what was done in run_uhi_analysis)
            # Add building coverage if available
            try:
                df_val = multi_buffer_building_coverage(df_val, "Building_Footprint.kml", [50,100,200])
            except Exception as e:
                print(f"Warning: Could not add building coverage: {e}")
            
            # Add spatial features
            df_val = add_spatial_features(df_val)
            
            # Get features and scaler from training
            df_train, _, _, features, scaler = run_uhi_analysis(return_features=True)
            
            # Evaluate on validation data
            val_results, val_metrics = evaluate_on_validation(model, df_val, features, scaler)
            
            # Export validation results with detailed metrics
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            val_xlsx = os.path.join(DATA_DIR, f"Validation_Results_{timestamp}.xlsx")
            
            with pd.ExcelWriter(val_xlsx, engine='openpyxl') as writer:
                val_results.to_excel(writer, sheet_name='Validation_Predictions', index=False)
                
                if val_metrics:
                    metrics_df = pd.DataFrame({
                        'Metric': list(val_metrics.keys()),
                        'Value': list(val_metrics.values())
                    })
                    metrics_df.to_excel(writer, sheet_name='Validation_Metrics', index=False)
            
            print(f"Validation results exported to {val_xlsx}")
            
            # Plot validation results
            plot_uhi_basemap(val_results, pred_col="Predicted_UHI")
            
            # If true values are available, plot residuals
            if "True_UHI" in val_results.columns:
                plot_uhi_basemap(val_results, pred_col="Residual")
    else:
        print("\nNo validation data found. Place a CSV file named 'validation_data.csv' in the data directory to test the model.")

if __name__ == "__main__":
    main()