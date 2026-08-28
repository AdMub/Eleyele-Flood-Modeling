import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create a folder for EDA if it doesn't exist
os.makedirs('EDA', exist_ok=True)

# 1. Load the Master Dataset
df = pd.read_csv('Eleyele_HydroMet_Master.csv')

# Ensure Date is the index
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
else:
    # Fallback if the index is already datetime but unnamed
    df.index = pd.to_datetime(df.index)

# Define column names based on your master dataset
rain_col = 'PRECTOTCORR'
flow_col = 'Q_peak_m3s' 

# 2. Setup the Matplotlib Canvas
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# =====================================================================
# PANEL A: Full 36-Year Timeseries (30-Day Rolling Average for clarity)
# =====================================================================
# Apply a 30-day moving average to smooth out the daily noise
rain_roll = df[rain_col].rolling(window=30, min_periods=1).mean()
flow_roll = df[flow_col].rolling(window=30, min_periods=1).mean()

# Plot Rainfall on primary Y-axis (inverted to look like falling rain, a standard in hydrology)
ax1.plot(df.index, rain_roll, color='dodgerblue', alpha=0.8, linewidth=1.5, label='Rainfall (30-Day Avg)')
ax1.set_ylabel('Rainfall (mm/day)', color='dodgerblue', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor='dodgerblue')

# Plot Streamflow on secondary Y-axis
ax1_twin = ax1.twinx()
ax1_twin.plot(df.index, flow_roll, color='crimson', alpha=0.9, linewidth=1.5, label='Peak Streamflow (30-Day Avg)')
ax1_twin.set_ylabel('Peak Discharge (m³/s)', color='crimson', fontsize=12, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor='crimson')

ax1.set_title('A. 36-Year Historical Hydrometeorological Trend', fontsize=14, fontweight='bold')

# Combine legends for Panel A
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_twin.get_legend_handles_labels()
ax1_twin.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, shadow=True)

# =====================================================================
# PANEL B: Hydrological Seasonality (Average Monthly Climatology)
# =====================================================================
# Group by month across the entire 36 years to get the seasonal pattern
monthly_avg = df.groupby(df.index.month)[[rain_col, flow_col]].mean()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot Average Rainfall as bars
ax2.bar(months, monthly_avg[rain_col], color='dodgerblue', alpha=0.6, edgecolor='black')
ax2.set_ylabel('Average Rainfall (mm/day)', color='dodgerblue', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='dodgerblue')

# Plot Average Streamflow as a line over the bars
ax2_twin = ax2.twinx()
ax2_twin.plot(months, monthly_avg[flow_col], color='crimson', marker='o', markersize=8, linewidth=3, label='Average Peak Streamflow')
ax2_twin.set_ylabel('Average Peak Discharge (m³/s)', color='crimson', fontsize=12, fontweight='bold')
ax2_twin.tick_params(axis='y', labelcolor='crimson')

ax2.set_title('B. Annual Seasonality (36-Year Monthly Averages)', fontsize=14, fontweight='bold')

# =====================================================================
# Final Polish and Export
# =====================================================================
plt.tight_layout(pad=3.0)
output_path = 'EDA/Figure_2_Seasonality.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Success! Figure 2 saved in high resolution (300 dpi) at: {output_path}")

plt.show()