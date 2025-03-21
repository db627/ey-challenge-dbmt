#!/usr/bin/env python

"""
create_submission.py

Usage:
  python create_submission.py

1) Checks if new_data.csv exists.
   - If not found, creates it with sample lat/lon rows.
2) Trains model via Main_Plot.py (run_uhi_analysis).
3) Preprocesses new_data.csv (building coverage, spatial features).
4) Generates predictions.
5) Saves a minimal Submission.csv with only: Longitude, Latitude, Predicted_UHI.
"""

import os
import numpy as np
import pandas as pd

# Import from your main script (adjust filename if not exactly "Main_Plot.py")
from Main_Plot import (
    run_uhi_analysis,
    multi_buffer_building_coverage,
    add_spatial_features,
    plot_uhi_basemap
)

def main():
    # ------------------------------------------------------------------
    # 1) If new_data.csv is missing, create it with sample data
    # ------------------------------------------------------------------
    sample_csv = "data/Submission_template_UHI2025-v2.csv"

    # ------------------------------------------------------------------
    # 2) Train (or load) the model from Main_Plot.py
    # ------------------------------------------------------------------
    print("Training/loading the model from Main_Plot.py...")
    df_train, r2_val, model, features, scaler = run_uhi_analysis(return_features=True)
    print(f"Model training complete. R² on test split: {r2_val:.4f}")

    # ------------------------------------------------------------------
    # 3) Load the new_data.csv
    # ------------------------------------------------------------------
    if not os.path.exists(sample_csv):
        print(f"ERROR: File not found -> {sample_csv}")
        return

    df_new = pd.read_csv(sample_csv)
    if "Longitude" not in df_new.columns or "Latitude" not in df_new.columns:
        print("ERROR: CSV must contain at least 'Longitude' and 'Latitude'.")
        return

    # ------------------------------------------------------------------
    # 4) Compute building coverage (optional) & add spatial features
    # ------------------------------------------------------------------
    try:
        df_new = multi_buffer_building_coverage(df_new, "Building_Footprint.kml", buffers=[50, 100, 200])
    except Exception as e:
        print(f"Warning: Could not compute building coverage: {e}")

    df_new = add_spatial_features(df_new)

    # ------------------------------------------------------------------
    # 5) Ensure all required features exist (fill missing with 0)
    # ------------------------------------------------------------------
    for feat in features:
        if feat not in df_new.columns:
            df_new[feat] = 0

    X_new = df_new[features].copy()

    # ------------------------------------------------------------------
    # 6) Scale and predict
    # ------------------------------------------------------------------
    X_new_scaled = scaler.transform(X_new)
    y_pred_new = model.predict(X_new_scaled)
    df_new["Predicted_UHI"] = y_pred_new

    # ------------------------------------------------------------------
    # 7) Save only the minimal columns to Submission.csv
    # ------------------------------------------------------------------
    out_csv = "Submission.csv"
    df_minimal = df_new[["Longitude", "Latitude", "Predicted_UHI"]]
    df_minimal.to_csv(out_csv, index=False)
    print('Saved to CSV')


    # 8) Plot the UHI basemap using the df_minimal dataframe (Longitude, Latitude, Predicted_UHI)
#    Make sure you've imported `plot_uhi_basemap` from Main_Plot at the top of create_submission.py:
#
#    from Main_Plot import plot_uhi_basemap
#
#    and have contextily (ctx) installed and available.
#

    plot_uhi_basemap(df_minimal, lat_col="Latitude", lon_col="Longitude", pred_col="Predicted_UHI")
if __name__ == "__main__":
    main()