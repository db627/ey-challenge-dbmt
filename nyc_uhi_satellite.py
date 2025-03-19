import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
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


def download_satellite_imagery():
    """
    Download satellite imagery from Microsoft Planetary Computer for NYC area
    """
    print("\nDownloading satellite imagery from Microsoft Planetary Computer...")
    
    aoi_bbox = [-74.03, 40.7, -73.85, 40.9]
    
    satellite_file = os.path.join(data_dir, 'nyc_satellite.tif')
    
    if os.path.exists(satellite_file):
        print(f"Satellite image already exists at {satellite_file}")
        return satellite_file
    
    try:
        catalog = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        
        print("Searching for Landsat 8/9 data (July 2021)...")
        search = catalog.search(
            collections=["landsat-c2-l2"],
            datetime="2021-07-20/2021-07-28",
            bbox=aoi_bbox,
            query={"eo:cloud_cover": {"lt": 20}}  
        )
        
        items = list(search.get_items())
        
        if not items:
            print("No suitable Landsat imagery found. Trying Sentinel-2...")
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                datetime="2021-07-20/2021-07-28",
                bbox=aoi_bbox,
                query={"eo:cloud_cover": {"lt": 30}}
            )
            items = list(search.get_items())
        
        if not items:
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
        
        items.sort(key=lambda item: item.properties.get('eo:cloud_cover', 100))
        item = items[0]
        
        print(f"Found suitable image: {item.id}")
        print(f"Date: {item.properties.get('datetime')}")
        print(f"Cloud cover: {item.properties.get('eo:cloud_cover')}%")
        print(f"Collection: {item.collection_id}")
        
        if item.collection_id == "landsat-c2-l2":
            visual_asset = item.assets.get('visual')
            if not visual_asset:
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
                
                rgb = xr.Dataset({
                    'red': red_data,
                    'green': green_data,
                    'blue': blue_data
                })
                
                red_data.rio.to_raster(satellite_file)
                
        elif item.collection_id == "sentinel-2-l2a":
            visual_asset = item.assets.get('visual')
            if not visual_asset:
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
    
    output_path = os.path.join(data_dir, 'nyc_synthetic_satellite.tif')
    
    if os.path.exists(output_path):
        print(f"Synthetic image already exists at {output_path}")
        return output_path
    
    try:
        uhi_data = pd.read_csv('trainingdata.csv')
        
        min_lon, max_lon = uhi_data['Longitude'].min(), uhi_data['Longitude'].max()
        min_lat, max_lat = uhi_data['Latitude'].min(), uhi_data['Latitude'].max()
        
        resolution = 0.0002  
        width = int((max_lon - min_lon) / resolution)
        height = int((max_lat - min_lat) / resolution)
        
        lon_grid = np.linspace(min_lon, max_lon, width)
        lat_grid = np.linspace(min_lat, max_lat, height)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Water (dark blue)
        rgb[:, :, 0] = 10   
        rgb[:, :, 1] = 20   
        rgb[:, :, 2] = 60   
        
        # Add land areas
        for i in range(height):
            for j in range(width):
                # Manhattan
                if (-74.02 < lon_mesh[i, j] < -73.95 and 
                    40.7 < lat_mesh[i, j] < 40.87):
                    rgb[i, j, 0] = 150  
                    rgb[i, j, 1] = 150  
                    rgb[i, j, 2] = 150  
                
                # Bronx
                if (-73.94 < lon_mesh[i, j] < -73.86 and 
                    40.81 < lat_mesh[i, j] < 40.89):
                    rgb[i, j, 0] = 160  
                    rgb[i, j, 1] = 160  
                    rgb[i, j, 2] = 160  
        
        # Central Park
        park_left = int(((-73.97 - min_lon) / (max_lon - min_lon)) * width)
        park_right = int(((-73.95 - min_lon) / (max_lon - min_lon)) * width)
        park_bottom = int(((40.77 - min_lat) / (max_lat - min_lat)) * height)
        park_top = int(((40.8 - min_lat) / (max_lat - min_lat)) * height)
        
        rgb[park_bottom:park_top, park_left:park_right, 0] = 30   
        rgb[park_bottom:park_top, park_left:park_right, 1] = 140  
        rgb[park_bottom:park_top, park_left:park_right, 2] = 30   
        
        # Harlem river
        harlem_x = np.linspace(park_right, width-1, 100).astype(int)
        harlem_y = np.linspace(park_top, height-1, 100).astype(int)
        for i in range(len(harlem_x) - 1):
            x, y = harlem_x[i], harlem_y[i]
            if 0 <= x < width and 0 <= y < height:
                rgb[y-5:y+5, x-2:x+2, 0] = 10  
                rgb[y-5:y+5, x-2:x+2, 1] = 20  
                rgb[y-5:y+5, x-2:x+2, 2] = 80  
        
        # Simulated streets
        for i in range(20, height, 40):
            thickness = 3 if i % 120 == 20 else 1
            rgb[i-thickness:i+thickness, :, 0] = 100
            rgb[i-thickness:i+thickness, :, 1] = 100
            rgb[i-thickness:i+thickness, :, 2] = 100
        
        for j in range(20, width, 40):
            thickness = 3 if j % 120 == 20 else 1
            rgb[:, j-thickness:j+thickness, 0] = 100
            rgb[:, j-thickness:j+thickness, 1] = 100
            rgb[:, j-thickness:j+thickness, 2] = 100
            
        transform = rasterio.transform.from_bounds(
            min_lon, min_lat, max_lon, max_lat, width, height
        )
        
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


def run_uhi_analysis():
    """Run the UHI analysis using random forest model"""
    print("\nRunning Urban Heat Island analysis...")
    
    uhi_data = pd.read_csv('trainingdata.csv')
    
    print(f"Dataset shape: {uhi_data.shape}")
    print("\nFirst 5 rows:")
    print(uhi_data.head())
    
    missing_values = uhi_data.isnull().sum()
    print("\nMissing values:")
    print(missing_values)
    
    uhi_data['datetime'] = pd.to_datetime(uhi_data['datetime'], format='%d-%m-%Y %H:%M')
    uhi_data['hour'] = uhi_data['datetime'].dt.hour
    uhi_data['minute'] = uhi_data['datetime'].dt.minute
    
    try:
        bronx_weather = pd.read_excel('NY_Mesonet_Weather.xlsx', sheet_name='Bronx')
        manhattan_weather = pd.read_excel('NY_Mesonet_Weather.xlsx', sheet_name='Manhattan')
        
        print("\nWeather data loaded successfully")
        
        bronx_nan = bronx_weather['Date / Time'].isna().sum()
        manhattan_nan = manhattan_weather['Date / Time'].isna().sum()
        
        if bronx_nan > 0:
            bronx_weather = bronx_weather.dropna(subset=['Date / Time'])
        
        if manhattan_nan > 0:
            manhattan_weather = manhattan_weather.dropna(subset=['Date / Time'])
        
        bronx_weather['Date / Time'] = pd.to_datetime(bronx_weather['Date / Time'], errors='coerce')
        manhattan_weather['Date / Time'] = pd.to_datetime(manhattan_weather['Date / Time'], errors='coerce')
        
        bronx_weather = bronx_weather.dropna(subset=['Date / Time'])
        manhattan_weather = manhattan_weather.dropna(subset=['Date / Time'])
        
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
        
        uhi_data['datetime_rounded'] = uhi_data['datetime'].dt.floor('5min')
        bronx_weather['datetime_rounded'] = bronx_weather['datetime'].dt.floor('5min')
        manhattan_weather['datetime_rounded'] = manhattan_weather['datetime'].dt.floor('5min')
        
        weather_combined = pd.merge(
            bronx_weather[['datetime_rounded', 'air_temp_bronx', 'humidity_bronx', 'wind_speed_bronx', 'wind_dir_bronx', 'solar_flux_bronx']], 
            manhattan_weather[['datetime_rounded', 'air_temp_manhattan', 'humidity_manhattan', 'wind_speed_manhattan', 'wind_dir_manhattan', 'solar_flux_manhattan']], 
            on='datetime_rounded'
        )
        
        weather_combined['avg_air_temp'] = (weather_combined['air_temp_bronx'] + weather_combined['air_temp_manhattan']) / 2
        weather_combined['avg_humidity'] = (weather_combined['humidity_bronx'] + weather_combined['humidity_manhattan']) / 2
        weather_combined['avg_wind_speed'] = (weather_combined['wind_speed_bronx'] + weather_combined['wind_speed_manhattan']) / 2
        weather_combined['avg_solar_flux'] = (weather_combined['solar_flux_bronx'] + weather_combined['solar_flux_manhattan']) / 2
        
        data_merged = pd.merge(uhi_data, weather_combined, on='datetime_rounded')
        
        print(f"Merged data shape: {data_merged.shape}")
        
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
    
    print("\nPerforming feature engineering...")
    
    if not use_weather:
        data_merged = uhi_data.copy()
        
        data_merged['minutes_since_noon'] = (data_merged['hour'] - 12) * 60 + data_merged['minute']
        data_merged['time_of_day'] = np.sin(2 * np.pi * (data_merged['hour'] * 60 + data_merged['minute']) / (24 * 60))
        
        features = ['hour', 'minute', 'minutes_since_noon', 'time_of_day']
        
    else:
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate the Haversine distance between two points on earth"""
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            r = 6371  # Radius of earth in kilometers
            return c * r
        
        data_merged['dist_to_bronx_station'] = data_merged.apply(
            lambda row: haversine_distance(row['Latitude'], row['Longitude'], 40.8679, -73.8806), axis=1
        )
        
        data_merged['dist_to_manhattan_station'] = data_merged.apply(
            lambda row: haversine_distance(row['Latitude'], row['Longitude'], 40.7893, -73.9631), axis=1
        )
        
        data_merged['minutes_since_noon'] = (data_merged['hour'] - 12) * 60 + data_merged['minute']
        data_merged['time_of_day'] = np.sin(2 * np.pi * (data_merged['hour'] * 60 + data_merged['minute']) / (24 * 60))
        
        data_merged['weighted_temp'] = (
            data_merged['air_temp_bronx'] / (data_merged['dist_to_bronx_station'] + 0.001) + 
            data_merged['air_temp_manhattan'] / (data_merged['dist_to_manhattan_station'] + 0.001)
        ) / (
            1 / (data_merged['dist_to_bronx_station'] + 0.001) + 
            1 / (data_merged['dist_to_manhattan_station'] + 0.001)
        )
        
        total_dist = data_merged['dist_to_bronx_station'] + data_merged['dist_to_manhattan_station']
        data_merged['manhattan_influence'] = data_merged['dist_to_bronx_station'] / total_dist
        
        data_merged['temp_solar_interaction'] = data_merged['avg_air_temp'] * data_merged['avg_solar_flux']
        data_merged['wind_solar_interaction'] = data_merged['avg_wind_speed'] * data_merged['avg_solar_flux']
        data_merged['temp_humidity_interaction'] = data_merged['avg_air_temp'] * data_merged['avg_humidity']
        
        features = [
            'hour', 'minute', 'minutes_since_noon', 'time_of_day',
            'air_temp_bronx', 'humidity_bronx', 'wind_speed_bronx', 'wind_dir_bronx', 'solar_flux_bronx',
            'air_temp_manhattan', 'humidity_manhattan', 'wind_speed_manhattan', 'wind_dir_manhattan', 'solar_flux_manhattan',
            'avg_air_temp', 'avg_humidity', 'avg_wind_speed', 'avg_solar_flux',
            'dist_to_bronx_station', 'dist_to_manhattan_station', 'weighted_temp', 'manhattan_influence',
            'temp_solar_interaction', 'wind_solar_interaction', 'temp_humidity_interaction'
        ]
    
    if 'datetime' in data_merged.columns:
        data_merged = data_merged.drop(['datetime'], axis=1)
    if 'datetime_rounded' in data_merged.columns:
        data_merged = data_merged.drop(['datetime_rounded'], axis=1)
    
    target = 'UHI Index'
    
    print(f"\nSelected {len(features)} features for modeling")
    
    features = [f for f in features if f in data_merged.columns]
    
    X = data_merged[features]
    y = data_merged[target]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    residuals = y_test - y_pred
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest Model Results:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    
    importances = rf_model.feature_importances_
    feature_names = np.array(features)
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 5 most important features:")
    for i in range(min(5, len(indices))):
        feature_idx = indices[i]
        print(f"  {feature_names[feature_idx]}: {importances[feature_idx]:.4f}")
    
    coords_test = data_merged.loc[y_test.index, ['Latitude', 'Longitude']]
    
    return {
        'coords_test': coords_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'residuals': residuals,
        'data_merged': data_merged
    }

def visualize_uhi_with_satellite(results, satellite_path):
    """Create visualization of predicted UHI data on satellite imagery"""
    print("\nCreating UHI visualization with satellite imagery...")
    
    coords_test = results['coords_test']
    y_test = results['y_test']
    y_pred = results['y_pred']
    data_merged = results['data_merged']
    
    uhi_cmap = LinearSegmentedColormap.from_list('uhi_cmap', 
                                                ['blue', 'green', 'yellow', 'orange', 'red'], 
                                                N=256)
    
    gdf = gpd.GeoDataFrame(
        coords_test.copy(),
        geometry=[Point(xy) for xy in zip(coords_test['Longitude'], coords_test['Latitude'])]
    )
    gdf['UHI_Index'] = y_test.values
    gdf['Predicted_UHI'] = y_pred
    
    gdf.crs = "EPSG:4326"
    
    satellite_available = satellite_path and os.path.exists(satellite_path)
    
    if satellite_available:
        try:
            with rasterio.open(satellite_path) as src:
                satellite_img = src.read()
                satellite_crs = src.crs
                
                print(f"Satellite image loaded: {satellite_path}")
                print(f"  Size: {src.width} x {src.height} pixels")
                print(f"  CRS: {satellite_crs}")
                
                # Create visualization for predicted UHI on satellite background
                fig, ax = plt.subplots(figsize=(15, 12))
                
                # Show the satellite image
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
                plt.savefig(os.path.join(fig_dir, "predicted_uhi_satellite.png"), dpi=300, bbox_inches='tight')
                plt.show()
                
        except Exception as e:
            print(f"Error creating satellite visualization: {e}")
            print("Falling back to standard visualization...")
            satellite_available = False
    
    # If satellite visualization failed or no satellite image is available
    if not satellite_available:
        print("Creating standard visualization without satellite imagery...")
        
        try:
            # Convert to Web Mercator for basemap compatibility
            gdf_web = gdf.to_crs(epsg=3857)
            
            ax = gdf_web.plot(
                column='Predicted_UHI',
                cmap=uhi_cmap,
                alpha=0.7,
                s=30,
                edgecolor='k',
                linewidth=0.5,
                figsize=(12, 10)
            )
            
            # Add basemap
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
            
            # Set title
            ax.set_title('Predicted Urban Heat Island Index with Basemap\n(Bronx and Manhattan)', fontsize=16)
            ax.set_axis_off()
            
            # Add colorbar
            norm = plt.Normalize(vmin=y_pred.min(), vmax=y_pred.max())
            sm = plt.cm.ScalarMappable(cmap=uhi_cmap, norm=norm)
            sm.set_array([])
            fig = ax.get_figure()
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Predicted UHI Index', fontsize=14)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, "predicted_uhi_basemap.png"), dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Error adding basemap: {e}")
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create scatter plot
            scatter = ax.scatter(
                coords_test['Longitude'], 
                coords_test['Latitude'],
                c=y_pred,
                cmap=uhi_cmap,
                alpha=0.8,
                s=30,
                edgecolor='k',
                linewidth=0.5
            )
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Predicted UHI Index', fontsize=14)
            
            # Add title and labels
            ax.set_title('Predicted NYC Urban Heat Island Index\n(Bronx and Manhattan)', fontsize=16)
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
            plt.savefig(os.path.join(fig_dir, "predicted_uhi_map.png"), dpi=300, bbox_inches='tight')
            plt.show()
    
    print(f"\nPredicted UHI visualization saved to {fig_dir}")
    return True


def main():
    
    satellite_path = download_satellite_imagery()
    
    
    results = run_uhi_analysis()
    
    
    visualize_uhi_with_satellite(results, satellite_path)
    
    print("\nAnalysis complete. Results and visualizations have been saved.")


if __name__ == "__main__":
    main()