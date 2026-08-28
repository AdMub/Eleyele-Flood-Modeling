import pandas as pd
import numpy as np
import os

print("Starting State-of-the-Art Data Fusion...")

# ---------------------------------------------------------
# 1. PROCESS STREAMFLOW (The Hydrological Component)
# ---------------------------------------------------------
print("Loading GEOGloWS streamflow...")
flow_path = 'Hydrological/Ona_River_Historical_Streamflow.csv'
df_flow = pd.read_csv(flow_path)

# Convert time to datetime object and set as index
df_flow['time'] = pd.to_datetime(df_flow['time'], utc=True)
df_flow.set_index('time', inplace=True)

# NOVELTY INJECTION: Don't just average the flow! 
# We extract Mean, Max (Peak Flood Pulse), and Sum (Total Volume)
daily_flow = df_flow.resample('D').agg({
    '140577794': ['mean', 'max', 'sum']
})
daily_flow.columns = ['Q_mean_m3s', 'Q_peak_m3s', 'Q_total_vol_proxy']

# Remove timezone to match NASA data
daily_flow.index = daily_flow.index.tz_localize(None).normalize()


# ---------------------------------------------------------
# 2. PROCESS METEOROLOGY (The Climate Component)
# ---------------------------------------------------------
print("Loading NASA POWER climate data...")
meteo_path = 'Meteorological/POWER_Point_Daily_19900101_20251231_007d40N_003d80E_LST.csv'

# Skip header lines dynamically
with open(meteo_path, 'r') as file:
    lines = file.readlines()
    skip_idx = 0
    for i, line in enumerate(lines):
        if "-END HEADER-" in line:
            skip_idx = i + 1
            break

df_meteo = pd.read_csv(meteo_path, skiprows=skip_idx)

# FIX: Parse the Date using YEAR and DOY (Day of Year)
if 'DOY' in df_meteo.columns and 'YEAR' in df_meteo.columns:
    # Combine Year and DOY (padded with zeros) into a string, then convert
    date_str = df_meteo['YEAR'].astype(str) + df_meteo['DOY'].astype(str).str.zfill(3)
    df_meteo['Date'] = pd.to_datetime(date_str, format='%Y%j')
else:
    # Fallback just in case
    df_meteo['Date'] = pd.to_datetime(df_meteo[['YEAR', 'MO', 'DY']].rename(columns={'YEAR':'year', 'MO':'month', 'DY':'day'}))

df_meteo.set_index('Date', inplace=True)


# ---------------------------------------------------------
# 3. MERGE DATABASES 
# ---------------------------------------------------------
print("Fusing datasets...")
# Merge on the date index. Keep dates where we have BOTH weather and flow data.
master_df = pd.merge(df_meteo, daily_flow, left_index=True, right_index=True, how='inner')


# ---------------------------------------------------------
# 4. PHYSICAL FEATURE ENGINEERING (To Satisfy Top Journals)
# ---------------------------------------------------------
print("Engineering hydrological memory features...")

# Calculate Antecedent Precipitation (how wet the soil is from previous days)
# Assuming 'PRECTOTCORR' is the corrected precipitation column
if 'PRECTOTCORR' in master_df.columns:
    rain_col = 'PRECTOTCORR'
    
    # 3-Day and 7-Day rolling rainfall totals (Hydrological Memory)
    master_df['Rain_3Day_Roll'] = master_df[rain_col].rolling(window=3).sum()
    master_df['Rain_7Day_Roll'] = master_df[rain_col].rolling(window=7).sum()
    
    # Fill the first few empty rolling rows with 0
    master_df.fillna(0, inplace=True)

# ---------------------------------------------------------
# 5. EXPORT MASTER DATASET
# ---------------------------------------------------------
output_file = 'Eleyele_HydroMet_Master.csv'
master_df.to_csv(output_file)
print(f"Success! Master dataset saved as: {output_file}")
print(f"Total modeling days available: {len(master_df)}")