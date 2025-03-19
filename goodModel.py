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
# 1) HELPER FUNCTIONS
###############################################################################

def load_uhi_data(uhi_csv="trainingdata.csv"):
    """
    Load the main UHI training data. 
    Returns a DataFrame with columns: Longitude, Latitude, datetime (as string), UHI Index
    """
    df = pd.read_csv(uhi_csv)
    # Parse datetime carefully; if your format is "24-07-2021 15:53", use dayfirst=True or a custom format
    # So:
    #   dayfirst=True OR format='%d-%m-%Y %H:%M'
    df['datetime'] = pd.to_datetime(df['datetime'], format='%d-%m-%Y %H:%M', errors='coerce')
    return df

def load_weather_data(weather_xlsx="NY_Mesonet_Weather.xlsx"):
    """
    Loads Bronx & Manhattan weather data from Excel, returns two DataFrames.
    If your sheet names differ, adjust them here.
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
    and is closest in time. 
    Returns a DataFrame with merged columns for that weather dataset. 
    If no row is within the tolerance, merges with NaN.

    :param max_time_diff: string or Timedelta defining the max difference 
                          (e.g. '5min', '10min', '15min', etc.)
    :param suffix: string to append to columns from weather_df 
    """
    # Sort both dataframes by time
    uhi_df = uhi_df.sort_values(by=time_col_uhi).copy()
    weather_df = weather_df.sort_values(by=time_col_wx).copy()

    # We'll store results in a list, then concat
    merged_rows = []

    # Convert max_time_diff into a Timedelta
    max_diff = pd.to_timedelta(max_time_diff)

    # We'll do a pointer approach
    j = 0
    w_times = weather_df[time_col_wx].values

    for i in range(len(uhi_df)):
        u_time = uhi_df.iloc[i][time_col_uhi]
        # Move pointer in weather_df so that w_times[j] is close to u_time
        while j < len(w_times)-1 and abs((w_times[j+1] - u_time)) < abs((w_times[j] - u_time)):
            j += 1
        # Now weather_df.iloc[j] is the closest in time
        time_diff = abs(weather_df.iloc[j][time_col_wx] - u_time)
        if time_diff <= max_diff:
            # within tolerance
            row_merged = {**uhi_df.iloc[i].to_dict(), **weather_df.iloc[j].to_dict()}
        else:
            # out of tolerance => fill weather fields with NaN
            row_merged = uhi_df.iloc[i].to_dict()
            for c in weather_df.columns:
                if c == time_col_wx:
                    continue
                row_merged[c] = np.nan
        merged_rows.append(row_merged)

    merged_df = pd.DataFrame(merged_rows)

    if suffix:
        # rename columns from weather if needed
        w_cols = [c for c in weather_df.columns if c != time_col_wx]
        rename_map = {c: c + suffix for c in w_cols}
        merged_df.rename(columns=rename_map, inplace=True)

    return merged_df

def combine_bronx_manhattan_weather(uhi_df, bronx_df, manhattan_df):
    """
    1. Nearest-time merge the UHI with Bronx, within ±10 minutes
    2. Nearest-time merge the result with Manhattan, within ±10 minutes
    3. Compute average columns
    """
    # Merge with Bronx first
    merged_bronx = nearest_time_merge(uhi_df, bronx_df, max_time_diff='10min')
    # Merge with Manhattan
    final_merged = nearest_time_merge(merged_bronx, manhattan_df, max_time_diff='10min', suffix='_man')

    # Compute average columns
    # Because we have e.g. 'air_temp_bronx' and 'air_temp_manhattan' as 'air_temp_manhattan' 
    # in the second merge -> Actually we appended _man suffix, so it's 'air_temp_manhattan_man'
    # So let's unify carefully:

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
    For each (lat/lon) row in df, compute building coverage fraction in multiple buffer distances.
    """
    try:
        gdf_buildings = gpd.read_file(building_kml)
        gdf_buildings = gdf_buildings.to_crs(epsg=2263)

        gdf_points = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
            crs="EPSG:4326"
        ).to_crs(epsg=2263)

        for buf_m in buffers:
            colname = f"bldg_cov_{buf_m}m"
            coverage_list = []
            # Buffer each point
            gdf_points["buffer_geom"] = gdf_points.geometry.buffer(buf_m)
            for idx, row in gdf_points.iterrows():
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
        return df  # fallback

def compute_ndvi_for_points(df, red_tif="nyc_satellite_red.tif", nir_tif="nyc_satellite_nir.tif"):
    """
    For each (lat/lon) row in df, compute NDVI from Red & NIR GeoTIFFs.
    """
    try:
        red_data = rioxarray.open_rasterio(os.path.join(DATA_DIR, red_tif))
        nir_data = rioxarray.open_rasterio(os.path.join(DATA_DIR, nir_tif))

        gdf_points = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
            crs="EPSG:4326"
        )

        if red_data.rio.crs != gdf_points.crs:
            gdf_points = gdf_points.to_crs(red_data.rio.crs)

        ndvi_vals = []
        for idx, row in gdf_points.iterrows():
            x, y = row.geometry.x, row.geometry.y
            red_pix = float(red_data.sel(x=x, y=y, method="nearest").values)
            nir_pix = float(nir_data.sel(x=x, y=y, method="nearest").values)
            if (nir_pix + red_pix) == 0:
                ndvi_vals.append(0.0)
            else:
                ndvi_vals.append((nir_pix - red_pix)/(nir_pix + red_pix))

        gdf_points["ndvi"] = ndvi_vals
        gdf_points.drop(columns="geometry", inplace=True)
        return pd.DataFrame(gdf_points)
    except Exception as e:
        print(f"NDVI computation failed: {e}")
        return df

def add_spatial_features(df):
    """
    Add polynomial expansions of lat/lon to capture spatial trends:
     - lat^2, lon^2, lat*lon
    """
    df["lat2"] = df["Latitude"]**2
    df["lon2"] = df["Longitude"]**2
    df["lat_lon"] = df["Latitude"] * df["Longitude"]
    return df

###############################################################################
# 2) MAIN MODELING FUNCTION
###############################################################################

def run_uhi_analysis():
    """
    End-to-end pipeline:
      1. Load UHI data
      2. Load weather data (Bronx & Manhattan), nearest-time merge
      3. Feature engineering: time, building coverage, NDVI, spatial expansions
      4. Optional polynomial feature expansions
      5. Random Forest with hyperparameter tuning
      6. Visualize results & export predictions
    """
    print("Loading UHI data...")
    df_uhi = load_uhi_data("trainingdata.csv")
    print(f"UHI shape: {df_uhi.shape}")

    # Basic time features
    df_uhi["hour"] = df_uhi["datetime"].dt.hour
    df_uhi["minute"] = df_uhi["datetime"].dt.minute

    # Add day of year, if you suspect multiple days
    df_uhi["day_of_year"] = df_uhi["datetime"].dt.dayofyear

    df_uhi["minutes_since_noon"] = (df_uhi["hour"] - 12)*60 + df_uhi["minute"]
    # cyclical time
    total_minutes_day = 24*60
    df_uhi["time_of_day"] = np.sin(2*np.pi*(df_uhi["hour"]*60 + df_uhi["minute"])/total_minutes_day)

    # Load weather
    try:
        bronx_df, manhattan_df = load_weather_data("NY_Mesonet_Weather.xlsx")
        # Merge them with nearest-time approach
        df_merged = combine_bronx_manhattan_weather(df_uhi, bronx_df, manhattan_df)
    except Exception as e:
        print(f"Weather data load/merge error: {e}")
        df_merged = df_uhi.copy()

    # Additional distance-based features if weather was successfully merged
    # (Check if we have columns like air_temp_bronx, etc.)
    if "air_temp_bronx" in df_merged.columns and "air_temp_manhattan_man" in df_merged.columns:
        # Haversine
        def haversine(lat1, lon1, lat2, lon2):
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371
            return c*r

        # stations
        bronx_lat, bronx_lon = 40.8679, -73.8806
        man_lat, man_lon     = 40.7893, -73.9631

        df_merged["dist_bronx"] = df_merged.apply(
            lambda r: haversine(r["Latitude"], r["Longitude"], bronx_lat, bronx_lon), axis=1
        )
        df_merged["dist_manh"] = df_merged.apply(
            lambda r: haversine(r["Latitude"], r["Longitude"], man_lat, man_lon), axis=1
        )

        # Weighted temp example
        if "air_temp_manhattan_man" in df_merged.columns:
            # rename for convenience
            df_merged["temp_bronx"] = df_merged["air_temp_bronx"]
            df_merged["temp_manh"]  = df_merged["air_temp_manhattan_man"]
            df_merged["weighted_temp"] = (
                df_merged["temp_bronx"]/(df_merged["dist_bronx"]+0.001) + 
                df_merged["temp_manh"]/(df_merged["dist_manh"]+0.001)
            ) / (
                1/(df_merged["dist_bronx"]+0.001) + 1/(df_merged["dist_manh"]+0.001)
            )

    # Building coverage
    df_merged = multi_buffer_building_coverage(df_merged, building_kml="Building_Footprint.kml", buffers=[50,100,200])

    # NDVI (requires red & nir tifs). If you have them, uncomment:
    # df_merged = compute_ndvi_for_points(df_merged, red_tif="nyc_satellite_red.tif", nir_tif="nyc_satellite_nir.tif")

    # Add lat/lon expansions
    df_merged = add_spatial_features(df_merged)

    # Drop any row missing the target
    target_col = "UHI Index"
    df_merged.dropna(subset=[target_col], inplace=True)

    # Potential Features
    candidate_feats = [
        "Latitude","Longitude","lat2","lon2","lat_lon",
        "hour","minute","day_of_year","minutes_since_noon","time_of_day",
        "air_temp_bronx","humidity_bronx","wind_speed_bronx","wind_dir_bronx","solar_flux_bronx",
        "air_temp_manhattan_man","humidity_manhattan_man","wind_speed_manhattan_man","wind_dir_manhattan_man","solar_flux_manhattan_man",
        "avg_air_temp","avg_humidity","avg_wind_speed","avg_solar_flux",
        "dist_bronx","dist_manh","weighted_temp",
        "bldg_cov_50m","bldg_cov_100m","bldg_cov_200m",
        "ndvi"
    ]
    # Keep only what's in df
    features = [f for f in candidate_feats if f in df_merged.columns]

    # Clean up (fill missing)
    for c in features:
        df_merged[c].fillna(df_merged[c].mean(), inplace=True)

    X = df_merged[features].copy()
    y = df_merged[target_col].copy()

    # Optional polynomial expansion of numeric features:
    # This can sometimes boost performance, but watch out for overfitting.
    # We'll pick a few features to expand, e.g. time_of_day, ndvi, bldg_cov_100m:
    poly_cols = ["time_of_day","bldg_cov_100m"]
    # Filter only those that exist
    poly_cols = [c for c in poly_cols if c in X.columns]

    if poly_cols:
        pf = PolynomialFeatures(degree=2, include_bias=False)
        subX = X[poly_cols].values
        subX_poly = pf.fit_transform(subX)
        poly_feature_names = pf.get_feature_names_out(poly_cols)
        # Create DataFrame
        df_poly = pd.DataFrame(subX_poly, columns=poly_feature_names, index=X.index)
        # Merge back
        X = pd.concat([X.drop(columns=poly_cols), df_poly], axis=1)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Random Forest hyperparam search
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

    # Visualization: actual vs. predicted
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5, edgecolor='k')
    plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],'r--')
    plt.title(f"Actual vs. Predicted UHI (R²={r2_val:.3f})")
    plt.xlabel("Actual UHI Index")
    plt.ylabel("Predicted UHI Index")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "RF_Actual_vs_Pred.png"), dpi=300)
    plt.show()

    # Feature importances
    importances = best_rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    sorted_feats = [X.columns[i] for i in sorted_idx]
    sorted_vals = importances[sorted_idx]

    plt.figure(figsize=(10,8))
    plt.barh(sorted_feats[::-1], sorted_vals[::-1])
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "RF_Feature_Importance.png"), dpi=300)
    plt.show()

    # Export predictions
    test_idx = y_test.index
    df_out = df_merged.loc[test_idx, ['Latitude','Longitude',target_col]].copy()
    df_out["Predicted_UHI"] = y_pred
    df_out["Residual"] = df_out[target_col] - df_out["Predicted_UHI"]
    out_xlsx = os.path.join(DATA_DIR, "Predictions.xlsx")
    df_out.to_excel(out_xlsx, index=False)
    print(f"Predictions exported to {out_xlsx}")

    return r2_val, df_out

###############################################################################
# 3) MAIN ENTRY POINT
###############################################################################

def main():
    r2, df_pred = run_uhi_analysis()
    print(f"\nFinal R²: {r2:.3f}")
    if r2 < 0.8:
        print("Keep refining features & data quality to push closer to 0.8+!")

if __name__ == "__main__":
    main()