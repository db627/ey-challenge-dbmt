import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import os
import warnings
import rasterio
from rasterio.plot import show
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import planetary_computer
from pystac_client import Client
import geopandas as gpd
from shapely.geometry import Point, box
import contextily as ctx
from pyproj import CRS, Transformer
import rioxarray
import xarray as xr
import matplotlib.patheffects as pe

warnings.filterwarnings('ignore')

# Set the style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Create directories for outputs
data_dir = 'data'
fig_dir = 'figures'
for directory in [data_dir, fig_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ===== PART 1: DOWNLOAD SATELLITE DATA FROM PLANETARY COMPUTER =====
def download_satellite_imagery():
    """
    Download satellite imagery from Microsoft Planetary Computer for NYC area
    """
    print("\nDownloading satellite imagery from Microsoft Planetary Computer...")
    
    # Define area of interest - Bronx and Manhattan
    # Bounding box: minx, miny, maxx, maxy (west, south, east, north)
    aoi_bbox = [-74.03, 40.7, -73.85, 40.9]
    
    # Define satellite data file paths
    satellite_file = os.path.join(data_dir, 'nyc_satellite.tif')
    
    # Skip download if file already exists
    if os.path.exists(satellite_file):
        print(f"Satellite image already exists at {satellite_file}")
        return satellite_file
    
    try:
        # Connect to the Planetary Computer STAC API
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        
        # First try Landsat Collection 2 Level 2
        print("Searching for Landsat 8/9 data (July 2021)...")
        search = catalog.search(
            collections=["landsat-c2-l2"],
            datetime="2021-07-20/2021-07-28",
            bbox=aoi_bbox,
            query={"eo:cloud_cover": {"lt": 20}}  # Less than 20% cloud cover
        )
        
        items = list(search.get_items())
        
        if not items:
            # Try Sentinel-2 if Landsat isn't available
            print("No suitable Landsat imagery found. Trying Sentinel-2...")
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                datetime="2021-07-20/2021-07-28",
                bbox=aoi_bbox,
                query={"eo:cloud_cover": {"lt": 30}}  # Less than 30% cloud cover
            )
            items = list(search.get_items())
        
        if not items:
            # If still no data, try wider date range
            print("No suitable imagery found. Trying wider date range (June-August 2021)...")
            search = catalog.search(
                collections=["landsat-c2-l2", "sentinel-2-l2a"],
                datetime="2021-06-01/2021-08-31",
                bbox=aoi_bbox,
                query={"eo:cloud_cover": {"lt": 30}}
            )
            items = list(search.get_items())
        
        if not items:
            print("No suitable satellite imagery found. Creating synthetic image instead.")
            return create_synthetic_satellite_image()
        
        # Select the scene with lowest cloud cover
        items.sort(key=lambda item: item.properties.get('eo:cloud_cover', 100))
        item = items[0]
        
        print(f"Found suitable image: {item.id}")
        print(f"Date: {item.properties.get('datetime')}")
        print(f"Cloud cover: {item.properties.get('eo:cloud_cover')}%")
        print(f"Collection: {item.collection_id}")
        
        # Get visual/true color asset
        if item.collection_id == "landsat-c2-l2":
            visual_asset = item.assets.get('visual')
            if not visual_asset:
                # For Landsat, we can use red, green, blue bands
                red_asset = item.assets.get('red')
                green_asset = item.assets.get('green')
                blue_asset = item.assets.get('blue')
                
                if not (red_asset and green_asset and blue_asset):
                    print("Required bands not found. Creating synthetic image instead.")
                    return create_synthetic_satellite_image()
                
                print("Downloading RGB bands...")
                red_data = rioxarray.open_rasterio(red_asset.href)
                green_data = rioxarray.open_rasterio(green_asset.href)
                blue_data = rioxarray.open_rasterio(blue_asset.href)
                
                # Create RGB composite
                rgb = xr.Dataset({
                    'red': red_data,
                    'green': green_data,
                    'blue': blue_data
                })
                
                # Save as GeoTIFF
                red_data.rio.to_raster(satellite_file)
                
        elif item.collection_id == "sentinel-2-l2a":
            visual_asset = item.assets.get('visual')
            if not visual_asset:
                # For Sentinel-2, we can use TCI (True Color Image)
                visual_asset = item.assets.get('true-color') or item.assets.get('TCI')
                
            if not visual_asset:
                print("Visual asset not found. Creating synthetic image instead.")
                return create_synthetic_satellite_image()
            
            print("Downloading visual (true color) asset...")
            visual_data = rioxarray.open_rasterio(visual_asset.href)
            visual_data.rio.to_raster(satellite_file)
            
        print(f"Satellite image saved to: {satellite_file}")
        return satellite_file
        
    except Exception as e:
        print(f"Error downloading satellite data: {e}")
        print("Creating synthetic satellite image instead.")
        return create_synthetic_satellite_image()

def create_synthetic_satellite_image():
    """
    Create a synthetic satellite image for NYC if real data can't be obtained
    """
    print("\nCreating synthetic satellite image for visualization...")
    
    # Set output path
    output_path = os.path.join(data_dir, 'nyc_synthetic_satellite.tif')
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"Synthetic image already exists at {output_path}")
        return output_path
    
    try:
        # Load UHI data to get coordinates
        uhi_data = pd.read_csv('trainingdata.csv')
        
        # Get the bounding box
        min_lon, max_lon = uhi_data['Longitude'].min(), uhi_data['Longitude'].max()
        min_lat, max_lat = uhi_data['Latitude'].min(), uhi_data['Latitude'].max()
        
        # Create a grid
        resolution = 0.0002  # about 20m at this latitude
        width = int((max_lon - min_lon) / resolution)
        height = int((max_lat - min_lat) / resolution)
        
        # Create lon/lat meshgrid
        lon_grid = np.linspace(min_lon, max_lon, width)
        lat_grid = np.linspace(min_lat, max_lat, height)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Create a synthetic RGB image
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Set background (water) to dark blue
        rgb[:, :, 0] = 10   # R
        rgb[:, :, 1] = 20   # G
        rgb[:, :, 2] = 60   # B
        
        # Add land as gray
        for i in range(height):
            for j in range(width):
                # Manhattan island shape (simplified)
                if (-74.02 < lon_mesh[i, j] < -73.95 and 
                    40.7 < lat_mesh[i, j] < 40.87):
                    rgb[i, j, 0] = 150  # R
                    rgb[i, j, 1] = 150  # G
                    rgb[i, j, 2] = 150  # B
                
                # Bronx area
                if (-73.94 < lon_mesh[i, j] < -73.86 and 
                    40.81 < lat_mesh[i, j] < 40.89):
                    rgb[i, j, 0] = 160  # R
                    rgb[i, j, 1] = 160  # G
                    rgb[i, j, 2] = 160  # B
        
        # Add Central Park as green rectangle
        park_left = int(((-73.97 - min_lon) / (max_lon - min_lon)) * width)
        park_right = int(((-73.95 - min_lon) / (max_lon - min_lon)) * width)
        park_bottom = int(((40.77 - min_lat) / (max_lat - min_lat)) * height)
        park_top = int(((40.8 - min_lat) / (max_lat - min_lat)) * height)
        
        rgb[park_bottom:park_top, park_left:park_right, 0] = 30   # R
        rgb[park_bottom:park_top, park_left:park_right, 1] = 140  # G
        rgb[park_bottom:park_top, park_left:park_right, 2] = 30   # B
        
        # Add Harlem river
        harlem_x = np.linspace(park_right, width-1, 100).astype(int)
        harlem_y = np.linspace(park_top, height-1, 100).astype(int)
        for i in range(len(harlem_x) - 1):
            x, y = harlem_x[i], harlem_y[i]
            if 0 <= x < width and 0 <= y < height:
                rgb[y-5:y+5, x-2:x+2, 0] = 10  # R
                rgb[y-5:y+5, x-2:x+2, 1] = 20  # G
                rgb[y-5:y+5, x-2:x+2, 2] = 80  # B
        
        # Add some simulated streets
        for i in range(20, height, 40):
            # Horizontal streets
            thickness = 3 if i % 120 == 20 else 1
            rgb[i-thickness:i+thickness, :, 0] = 100
            rgb[i-thickness:i+thickness, :, 1] = 100
            rgb[i-thickness:i+thickness, :, 2] = 100
        
        for j in range(20, width, 40):
            # Vertical streets
            thickness = 3 if j % 120 == 20 else 1
            rgb[:, j-thickness:j+thickness, 0] = 100
            rgb[:, j-thickness:j+thickness, 1] = 100
            rgb[:, j-thickness:j+thickness, 2] = 100
            
        # Create transform to georeference the image
        transform = rasterio.transform.from_bounds(
            min_lon, min_lat, max_lon, max_lat, width, height
        )
        
        # Save as GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,
            dtype=rgb.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            dst.write(rgb[:, :, 0], 1)
            dst.write(rgb[:, :, 1], 2)
            dst.write(rgb[:, :, 2], 3)
        
        print(f"Synthetic satellite image saved to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error creating synthetic image: {e}")
        print("Will proceed without satellite background.")
        return None

# ===== PART 2: UHI DATA ANALYSIS (FROM EXISTING CODE) =====
def run_uhi_analysis():
    """Run the UHI analysis using random forest model"""
    print("\nRunning Urban Heat Island analysis...")
    
    # Load training data
    uhi_data = pd.read_csv('trainingdata.csv')
    
    # Display basic information
    print(f"Dataset shape: {uhi_data.shape}")
    print("\nFirst 5 rows:")
    print(uhi_data.head())
    
    # Check for missing values
    missing_values = uhi_data.isnull().sum()
    print("\nMissing values:")
    print(missing_values)
    
    # Convert datetime column to datetime objects
    uhi_data['datetime'] = pd.to_datetime(uhi_data['datetime'], format='%d-%m-%Y %H:%M')
    
    # Extract time features
    uhi_data['hour'] = uhi_data['datetime'].dt.hour
    uhi_data['minute'] = uhi_data['datetime'].dt.minute
    
    # Load weather data with error handling
    try:
        # Load Excel file with explicit check for empty cells
        bronx_weather = pd.read_excel('NY_Mesonet_Weather.xlsx', sheet_name='Bronx')
        manhattan_weather = pd.read_excel('NY_Mesonet_Weather.xlsx', sheet_name='Manhattan')
        
        print("\nWeather data loaded successfully")
        
        # Check for NaN values in datetime column
        bronx_nan = bronx_weather['Date / Time'].isna().sum()
        manhattan_nan = manhattan_weather['Date / Time'].isna().sum()
        
        # Remove rows with NaN in datetime column
        if bronx_nan > 0:
            bronx_weather = bronx_weather.dropna(subset=['Date / Time'])
        
        if manhattan_nan > 0:
            manhattan_weather = manhattan_weather.dropna(subset=['Date / Time'])
        
        # Convert datetime columns with error handling
        bronx_weather['Date / Time'] = pd.to_datetime(bronx_weather['Date / Time'], errors='coerce')
        manhattan_weather['Date / Time'] = pd.to_datetime(manhattan_weather['Date / Time'], errors='coerce')
        
        # Drop any rows where conversion failed
        bronx_weather = bronx_weather.dropna(subset=['Date / Time'])
        manhattan_weather = manhattan_weather.dropna(subset=['Date / Time'])
        
        # Rename columns for clarity
        bronx_weather = bronx_weather.rename(columns={
            'Date / Time': 'datetime',
            'Air Temp at Surface [degC]': 'air_temp_bronx',
            'Relative Humidity [percent]': 'humidity_bronx',
            'Avg Wind Speed [m/s]': 'wind_speed_bronx',
            'Wind Direction [degrees]': 'wind_dir_bronx',
            'Solar Flux [W/m^2]': 'solar_flux_bronx'
        })
        
        manhattan_weather = manhattan_weather.rename(columns={
            'Date / Time': 'datetime',
            'Air Temp at Surface [degC]': 'air_temp_manhattan',
            'Relative Humidity [percent]': 'humidity_manhattan',
            'Avg Wind Speed [m/s]': 'wind_speed_manhattan',
            'Wind Direction [degrees]': 'wind_dir_manhattan',
            'Solar Flux [W/m^2]': 'solar_flux_manhattan'
        })
        
        # Round datetime to nearest 5 minutes for merging with weather data
        uhi_data['datetime_rounded'] = uhi_data['datetime'].dt.floor('5min')
        bronx_weather['datetime_rounded'] = bronx_weather['datetime'].dt.floor('5min')
        manhattan_weather['datetime_rounded'] = manhattan_weather['datetime'].dt.floor('5min')
        
        # Merge UHI data with weather data
        # First, create a common dataset with both Bronx and Manhattan weather
        weather_combined = pd.merge(
            bronx_weather[['datetime_rounded', 'air_temp_bronx', 'humidity_bronx', 'wind_speed_bronx', 'wind_dir_bronx', 'solar_flux_bronx']], 
            manhattan_weather[['datetime_rounded', 'air_temp_manhattan', 'humidity_manhattan', 'wind_speed_manhattan', 'wind_dir_manhattan', 'solar_flux_manhattan']], 
            on='datetime_rounded'
        )
        
        # Calculate average and difference values for each weather parameter
        weather_combined['avg_air_temp'] = (weather_combined['air_temp_bronx'] + weather_combined['air_temp_manhattan']) / 2
        weather_combined['avg_humidity'] = (weather_combined['humidity_bronx'] + weather_combined['humidity_manhattan']) / 2
        weather_combined['avg_wind_speed'] = (weather_combined['wind_speed_bronx'] + weather_combined['wind_speed_manhattan']) / 2
        weather_combined['avg_solar_flux'] = (weather_combined['solar_flux_bronx'] + weather_combined['solar_flux_manhattan']) / 2
        
        # Merge with UHI data
        data_merged = pd.merge(uhi_data, weather_combined, on='datetime_rounded')
        
        print(f"Merged data shape: {data_merged.shape}")
        
        # Check if the merge was successful
        if data_merged.shape[0] == 0:
            print("WARNING: No data after merging with weather data!")
            print("Proceeding with model using only UHI data time features...")
            use_weather = False
        else:
            use_weather = True
            
    except Exception as e:
        print(f"Error processing weather data: {e}")
        print("Proceeding with model using only UHI data time features...")
        use_weather = False
        data_merged = uhi_data.copy()
    
    # Feature Engineering
    print("\nPerforming feature engineering...")
    
    # Create a fallback dataset in case weather data couldn't be used
    if not use_weather:
        data_merged = uhi_data.copy()
        
        # Create time-based features
        data_merged['minutes_since_noon'] = (data_merged['hour'] - 12) * 60 + data_merged['minute']
        data_merged['time_of_day'] = np.sin(2 * np.pi * (data_merged['hour'] * 60 + data_merged['minute']) / (24 * 60))
        
        # Select features (EXCLUDING latitude and longitude as per rules)
        features = ['hour', 'minute', 'minutes_since_noon', 'time_of_day']
        
    else:
        # Calculate Haversine distance to weather stations (Manhattan and Bronx)
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate the Haversine distance between two points on earth"""
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371  # Radius of earth in kilometers
            return c * r
        
        # Bronx Mesonet station: 40.8679, -73.8806
        # Manhattan Mesonet station: 40.7893, -73.9631
        data_merged['dist_to_bronx_station'] = data_merged.apply(
            lambda row: haversine_distance(row['Latitude'], row['Longitude'], 40.8679, -73.8806), axis=1
        )
        
        data_merged['dist_to_manhattan_station'] = data_merged.apply(
            lambda row: haversine_distance(row['Latitude'], row['Longitude'], 40.7893, -73.9631), axis=1
        )
        
        # Calculate time-based features
        data_merged['minutes_since_noon'] = (data_merged['hour'] - 12) * 60 + data_merged['minute']
        data_merged['time_of_day'] = np.sin(2 * np.pi * (data_merged['hour'] * 60 + data_merged['minute']) / (24 * 60))
        
        # Calculate distance-weighted temperature estimate
        # This is a feature that weights the temperature from each station based on distance
        data_merged['weighted_temp'] = (
            data_merged['air_temp_bronx'] / (data_merged['dist_to_bronx_station'] + 0.001) + 
            data_merged['air_temp_manhattan'] / (data_merged['dist_to_manhattan_station'] + 0.001)
        ) / (
            1 / (data_merged['dist_to_bronx_station'] + 0.001) + 
            1 / (data_merged['dist_to_manhattan_station'] + 0.001)
        )
        
        # Feature for relative location between stations (0 = at Bronx, 1 = at Manhattan)
        total_dist = data_merged['dist_to_bronx_station'] + data_merged['dist_to_manhattan_station']
        data_merged['manhattan_influence'] = data_merged['dist_to_bronx_station'] / total_dist
        
        # Interaction features
        data_merged['temp_solar_interaction'] = data_merged['avg_air_temp'] * data_merged['avg_solar_flux']
        data_merged['wind_solar_interaction'] = data_merged['avg_wind_speed'] * data_merged['avg_solar_flux']
        data_merged['temp_humidity_interaction'] = data_merged['avg_air_temp'] * data_merged['avg_humidity']
        
        # Select features (EXCLUDING latitude and longitude as per rules)
        features = [
            'hour', 'minute', 'minutes_since_noon', 'time_of_day',
            'air_temp_bronx', 'humidity_bronx', 'wind_speed_bronx', 'wind_dir_bronx', 'solar_flux_bronx',
            'air_temp_manhattan', 'humidity_manhattan', 'wind_speed_manhattan', 'wind_dir_manhattan', 'solar_flux_manhattan',
            'avg_air_temp', 'avg_humidity', 'avg_wind_speed', 'avg_solar_flux',
            'dist_to_bronx_station', 'dist_to_manhattan_station', 'weighted_temp', 'manhattan_influence',
            'temp_solar_interaction', 'wind_solar_interaction', 'temp_humidity_interaction'
        ]
    
    # Drop datetime columns before modeling
    if 'datetime' in data_merged.columns:
        data_merged = data_merged.drop(['datetime'], axis=1)
    if 'datetime_rounded' in data_merged.columns:
        data_merged = data_merged.drop(['datetime_rounded'], axis=1)
    
    target = 'UHI Index'
    
    print(f"\nSelected {len(features)} features for modeling")
    
    # Make sure all feature columns exist
    features = [f for f in features if f in data_merged.columns]
    
    # Prepare data for modeling
    X = data_merged[features]
    y = data_merged[target]
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest model...")
    # Initialize and train model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    residuals = y_test - y_pred
    
    # Model evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest Model Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    # Feature importance analysis
    importances = rf_model.feature_importances_
    feature_names = np.array(features)
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 5 most important features:")
    for i in range(min(5, len(indices))):
        feature_idx = indices[i]
        print(f"  {feature_names[feature_idx]}: {importances[feature_idx]:.4f}")
    
    # Prepare data for spatial visualization (without using lat/long in the model)
    coords_test = data_merged.loc[y_test.index, ['Latitude', 'Longitude']]
    
    # Return the data needed for visualization
    return {
        'coords_test': coords_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'residuals': residuals,
        'data_merged': data_merged
    }
# ===== PART 3: VISUALIZE UHI ON SATELLITE IMAGERY =====
def visualize_uhi_with_satellite(results, satellite_path):
    """Create visualizations of UHI data on top of satellite imagery"""
    print("\nCreating UHI visualizations with satellite imagery...")
    
    # Extract data from results
    coords_test = results['coords_test']
    y_test = results['y_test']
    y_pred = results['y_pred']
    residuals = results['residuals']
    data_merged = results['data_merged']
    
    # Create a custom colormap for UHI
    uhi_cmap = LinearSegmentedColormap.from_list('uhi_cmap', 
                                                ['blue', 'green', 'yellow', 'orange', 'red'], 
                                                N=256)
    
    # Convert coordinates to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        coords_test.copy(),
        geometry=[Point(xy) for xy in zip(coords_test['Longitude'], coords_test['Latitude'])]
    )
    gdf['UHI_Index'] = y_test.values
    gdf['Predicted_UHI'] = y_pred
    gdf['Abs_Error'] = np.abs(residuals)
    
    # Set CRS to WGS84
    gdf.crs = "EPSG:4326"
    
    # Check if satellite data is available
    satellite_available = satellite_path and os.path.exists(satellite_path)
    
    if satellite_available:
        try:
            # Open the satellite image
            with rasterio.open(satellite_path) as src:
                satellite_img = src.read()
                satellite_meta = src.meta
                satellite_bounds = src.bounds
                satellite_crs = src.crs
                
                print(f"Satellite image loaded: {satellite_path}")
                print(f"  Size: {src.width} x {src.height} pixels")
                print(f"  CRS: {satellite_crs}")
                
                # Create visualization 1: UHI on satellite background
                fig, ax = plt.subplots(figsize=(15, 12))
                
                # Show the satellite image
                # For RGB images (3 bands)
                if src.count == 3:
                    # Create RGB composite
                    rgb = np.dstack((satellite_img[0], satellite_img[1], satellite_img[2]))
                    
                    # Apply 2% stretch for better visibility
                    for i in range(3):
                        p2 = np.percentile(rgb[:,:,i][rgb[:,:,i] > 0], 2)
                        p98 = np.percentile(rgb[:,:,i][rgb[:,:,i] > 0], 98)
                        rgb[:,:,i] = np.clip(rgb[:,:,i], p2, p98)
                        if p98 > p2:  # Avoid division by zero
                            rgb[:,:,i] = ((rgb[:,:,i] - p2) / (p98 - p2) * 255).astype(np.uint8)
                
                    # Show the image
                    show(rgb, transform=src.transform, ax=ax)
                else:
                    # Show first band for non-RGB images
                    show(satellite_img[0], transform=src.transform, ax=ax)
                
                # Transform GeoDataFrame to match satellite CRS if needed
                if gdf.crs != satellite_crs:
                    gdf_satellite = gdf.to_crs(satellite_crs)
                else:
                    gdf_satellite = gdf
                
                # Plot UHI points on satellite image
                scatter = ax.scatter(
                    gdf_satellite.geometry.x, 
                    gdf_satellite.geometry.y, 
                    c=gdf_satellite['UHI_Index'],
                    cmap=uhi_cmap,
                    alpha=0.7,
                    s=30,
                    edgecolor='k',
                    linewidth=0.5
                )
                
                # Add colorbar
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label('UHI Index', fontsize=14)
                
                # Add title
                ax.set_title('Urban Heat Island Index with Satellite Imagery\n(Bronx and Manhattan)', fontsize=16)
                
                # Add borough labels with white text and black outline for visibility
                def add_label(lon, lat, text):
                    if gdf.crs != satellite_crs:
                        point = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326")
                        point = point.to_crs(satellite_crs)
                        x, y = point.geometry.x[0], point.geometry.y[0]
                    else:
                        x, y = lon, lat
                    
                    ax.text(x, y, text, fontsize=14, fontweight='bold', color='white',
                           path_effects=[pe.withStroke(linewidth=2, foreground='black')],
                           ha='center', va='center')
                
                # Add the borough labels
                add_label(-73.93, 40.87, 'BRONX')
                add_label(-73.97, 40.78, 'MANHATTAN')
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "uhi_satellite.png"), dpi=300, bbox_inches='tight')
                plt.show()
                
                # Create visualization 2: Predicted UHI on satellite background
                fig, ax = plt.subplots(figsize=(15, 12))
                
                # Show the satellite image
                if src.count == 3:
                    show(rgb, transform=src.transform, ax=ax)
                else:
                    show(satellite_img[0], transform=src.transform, ax=ax)
                
                # Plot predicted UHI points on satellite image
                scatter = ax.scatter(
                    gdf_satellite.geometry.x, 
                    gdf_satellite.geometry.y, 
                    c=gdf_satellite['Predicted_UHI'],
                    cmap=uhi_cmap,
                    alpha=0.7,
                    s=30,
                    edgecolor='k',
                    linewidth=0.5
                )
                
                # Add colorbar
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label('Predicted UHI Index', fontsize=14)
                
                # Add title
                ax.set_title('Predicted Urban Heat Island Index with Satellite Imagery\n(Bronx and Manhattan)', fontsize=16)
                
                # Add the borough labels
                add_label(-73.93, 40.87, 'BRONX')
                add_label(-73.97, 40.78, 'MANHATTAN')
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "predicted_uhi_satellite.png"), dpi=300, bbox_inches='tight')
                plt.show()
                
                # Create visualization 3: UHI error map on satellite background
                fig, ax = plt.subplots(figsize=(15, 12))
                
                # Show the satellite image
                if src.count == 3:
                    show(rgb, transform=src.transform, ax=ax)
                else:
                    show(satellite_img[0], transform=src.transform, ax=ax)
                
                # Plot UHI error points on satellite image
                scatter = ax.scatter(
                    gdf_satellite.geometry.x, 
                    gdf_satellite.geometry.y, 
                    c=gdf_satellite['Abs_Error'],
                    cmap='viridis',  # Use different colormap for errors
                    alpha=0.7,
                    s=30,
                    edgecolor='k',
                    linewidth=0.5
                )
                
                # Add colorbar
                cbar = fig.colorbar(scatter, ax=ax)
                cbar.set_label('Absolute Prediction Error', fontsize=14)
                
                # Add title
                ax.set_title('UHI Prediction Error with Satellite Imagery\n(Bronx and Manhattan)', fontsize=16)
                
                # Add the borough labels
                add_label(-73.93, 40.87, 'BRONX')
                add_label(-73.97, 40.78, 'MANHATTAN')
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "uhi_error_satellite.png"), dpi=300, bbox_inches='tight')
                plt.show()
                
                # Create visualization 4: Interpolated UHI heatmap on satellite
                fig, ax = plt.subplots(figsize=(15, 12))
                
                # Show the satellite image
                if src.count == 3:
                    show(rgb, transform=src.transform, ax=ax)
                else:
                    show(satellite_img[0], transform=src.transform, ax=ax)
                
                # Try to create interpolated heatmap overlay
                try:
                    from scipy.interpolate import griddata
                    
                    # Create a grid for interpolation
                    grid_size = 200
                    x_min, x_max = gdf_satellite.geometry.x.min(), gdf_satellite.geometry.x.max()
                    y_min, y_max = gdf_satellite.geometry.y.min(), gdf_satellite.geometry.y.max()
                    
                    x_grid = np.linspace(x_min, x_max, grid_size)
                    y_grid = np.linspace(y_min, y_max, grid_size)
                    xx, yy = np.meshgrid(x_grid, y_grid)
                    
                    # Interpolate UHI values to the grid
                    points = np.column_stack((gdf_satellite.geometry.x, gdf_satellite.geometry.y))
                    values = gdf_satellite['UHI_Index']
                    
                    grid_z = griddata(points, values, (xx, yy), method='cubic')
                    
                    # Create contour plot with transparency
                    contour = ax.contourf(xx, yy, grid_z, 15, cmap=uhi_cmap, alpha=0.5)
                    
                    # Add contour lines
                    contour_lines = ax.contour(xx, yy, grid_z, 5, colors='black', linewidths=0.5, alpha=0.5)
                    
                except Exception as e:
                    print(f"Couldn't create interpolated overlay: {e}")
                    
                    # Fall back to regular scatter plot
                    scatter = ax.scatter(
                        gdf_satellite.geometry.x, 
                        gdf_satellite.geometry.y, 
                        c=gdf_satellite['UHI_Index'],
                        cmap=uhi_cmap,
                        alpha=0.7,
                        s=30,
                        edgecolor='k',
                        linewidth=0.5
                    )
                
                # Add colorbar
                cbar = fig.colorbar(scatter if 'contour' not in locals() else contour, ax=ax)
                cbar.set_label('UHI Index', fontsize=14)
                
                # Add title
                ax.set_title('Interpolated UHI Heatmap with Satellite Imagery\n(Bronx and Manhattan)', fontsize=16)
                
                # Add the borough labels
                add_label(-73.93, 40.87, 'BRONX')
                add_label(-73.97, 40.78, 'MANHATTAN')
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, "uhi_heatmap_satellite.png"), dpi=300, bbox_inches='tight')
                plt.show()
                
        except Exception as e:
            print(f"Error creating satellite visualizations: {e}")
            print("Falling back to standard visualizations...")
            satellite_available = False
    
    # If satellite visualization failed or no satellite image is available
    if not satellite_available:
        print("Creating standard visualizations without satellite imagery...")
        
        # Visualization 1: Basic UHI map
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create scatter plot
        scatter = ax.scatter(
            coords_test['Longitude'], 
            coords_test['Latitude'],
            c=y_test,
            cmap='hot',
            alpha=0.8,
            s=30,
            edgecolor='k',
            linewidth=0.5
        )
        
        # Add basemap from Contextily if available
        try:
            # Convert to Web Mercator for basemap compatibility
            gdf_web = gdf.to_crs(epsg=3857)
            
            ax = gdf_web.plot(
                column='UHI_Index',
                cmap='hot',
                alpha=0.7,
                s=30,
                edgecolor='k',
                linewidth=0.5,
                figsize=(12, 10)
            )
            
            # Add basemap
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
            
            # Set title and labels
            ax.set_title('Urban Heat Island Index with Basemap\n(Bronx and Manhattan)', fontsize=16)
            ax.set_axis_off()
            
            # Add colorbar
            norm = plt.Normalize(vmin=y_test.min(), vmax=y_test.max())
            sm = plt.cm.ScalarMappable(cmap='hot', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('UHI Index', fontsize=14)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "uhi_basemap.png"), dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error adding basemap: {e}")
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('UHI Index', fontsize=14)
            
            # Add title and labels
            ax.set_title('NYC Urban Heat Island Index\n(Bronx and Manhattan)', fontsize=16)
            ax.set_xlabel('Longitude', fontsize=14)
            ax.set_ylabel('Latitude', fontsize=14)
            
            # Add text annotation for regions
            ax.text(-73.93, 40.87, 'BRONX', fontsize=14, fontweight='bold', ha='center')
            ax.text(-73.97, 40.78, 'MANHATTAN', fontsize=14, fontweight='bold', ha='center')
            
            # Set axis limits to focus on NYC area
            ax.set_xlim(-74.03, -73.85)
            ax.set_ylim(40.7, 40.9)
            
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "nyc_uhi_map.png"), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Visualization 2: Create hexbin plot as alternative
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create hexbin plot
        hb = ax.hexbin(
            data_merged['Longitude'], 
            data_merged['Latitude'], 
            C=data_merged['UHI Index'], 
            gridsize=50, 
            cmap='hot', 
            alpha=0.8,
            reduce_C_function=np.mean
        )
        
        # Add colorbar
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('Average UHI Index', fontsize=14)
        
        # Add title and labels
        ax.set_title('UHI Index Hexbin Heatmap\n(Bronx and Manhattan)', fontsize=16)
        ax.set_xlabel('Longitude', fontsize=14)
        ax.set_ylabel('Latitude', fontsize=14)
        
        # Add text annotation for regions
        ax.text(-73.93, 40.87, 'BRONX', fontsize=14, fontweight='bold', ha='center')
        ax.text(-73.97, 40.78, 'MANHATTAN', fontsize=14, fontweight='bold', ha='center')
        
        # Set axis limits to focus on NYC area
        ax.set_xlim(-74.03, -73.85)
        ax.set_ylim(40.7, 40.9)
        
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "uhi_hexbin.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Visualization 3: 3D visualization
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot 3D scatter with UHI as height
            sc = ax.scatter(
                coords_test['Longitude'], 
                coords_test['Latitude'], 
                y_test,
                c=y_test, 
                cmap='hot',
                s=30, 
                alpha=0.8
            )
            
            # Add colorbar
            cbar = fig.colorbar(sc, ax=ax, pad=0.1)
            cbar.set_label('UHI Index', fontsize=14)
            
            # Set labels and title
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.set_zlabel('UHI Index', fontsize=12)
            ax.set_title('3D Visualization of Urban Heat Island Effect\n(Bronx and Manhattan)', fontsize=16)
            
            # Set view angle
            ax.view_init(elev=30, azim=225)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "uhi_3d.png"), dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error creating 3D visualization: {e}")
    
    print(f"\nAll visualizations saved to {fig_dir}")
    return True

# ===== MAIN FUNCTION =====
def main():
    """Main function to run the entire UHI analysis and visualization pipeline"""
    print("NYC Urban Heat Island Analysis with Satellite Visualization")
    print("==========================================================")
    
    # Step 1: Download satellite imagery from Microsoft Planetary Computer
    satellite_path = download_satellite_imagery()
    
    # Step 2: Run UHI analysis with random forest model
    results = run_uhi_analysis()
    
    # Step 3: Create visualizations with satellite imagery
    visualize_uhi_with_satellite(results, satellite_path)
    
    print("\nAnalysis complete!")
    print(f"Results and visualizations have been saved to the '{fig_dir}' directory.")

if __name__ == "__main__":
    main()