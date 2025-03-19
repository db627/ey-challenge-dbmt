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

def run_uhi_analysis():
    """
    1) Load & merge data
    2) Feature engineering
    3) Train RandomForest with hyperparam search
    4) Return the final DataFrame w/ predictions
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

    # Export predictions
    out_xlsx = os.path.join(DATA_DIR, "Predictions.xlsx")
    df_out.to_excel(out_xlsx, index=False)
    print(f"\nPredictions exported to {out_xlsx}")

    return df_out, r2_val, best_rf

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
    ax.set_title("Predicted Urban Heat Island Index with Basemap\n(Bronx and Manhattan)", fontsize=16)

    # Add colorbar
    vmin, vmax = gdf_web[pred_col].min(), gdf_web[pred_col].max()
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=uhi_cmap)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Predicted UHI Index", fontsize=14)

    plt.tight_layout()
    out_fig = os.path.join(FIG_DIR, "predicted_uhi_basemap.png")
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

if __name__ == "__main__":
    main()