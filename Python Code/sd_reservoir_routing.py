import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('System_Dynamics', exist_ok=True)

print("--- Step 1: Loading IDF Synthetic Design Storms ---")
storms_df = pd.read_csv('Hydrological/Design_Storm_Hyetographs.csv')
return_periods = [5, 10, 20, 50]

# =====================================================================
# --- Step 2: Physical Parameters & Hydrological Constants ---
# =====================================================================
CATCHMENT_AREA_M2 = 325_000_000  
RESERVOIR_CAPACITY_M3 = 7_040_000 
SURFACE_AREA_M2 = 1_500_000      
SPILLWAY_LENGTH_M = 20.0         
WEIR_COEF = 1.7                  
BASEFLOW_M3S = 2.0

# Defensible Catchment Lag (Kirpich 1940 Formula)
L_METERS = 30000.0  
S_SLOPE = 0.002     
tc_minutes = 0.0195 * (L_METERS ** 0.77) * (S_SLOPE ** -0.385)
K_LAG_HOURS = tc_minutes / 60.0 
print(f"Calculated Kirpich Time of Concentration: {K_LAG_HOURS:.2f} hours")

SUB_STEPS_PER_HOUR = 12
DT_SECONDS = 3600.0 / SUB_STEPS_PER_HOUR

# SENSITIVITY MATRIX: Runoff Coefficients and Initial Storage
runoff_coefficients = [0.40, 0.50, 0.60]
initial_storage_percentages = [0.70, 0.90]

print("--- Step 2: Running Level-Pool SD Routing with Sensitivity Analysis ---")

results = []

for initial_pct in initial_storage_percentages:
    INITIAL_STORAGE = RESERVOIR_CAPACITY_M3 * initial_pct
    
    for runoff_coef in runoff_coefficients:
        for T in return_periods:
            rain_mm_hr = storms_df[f'{T}_yr_Rain_mm'].values
            
            storage = INITIAL_STORAGE
            catchment_runoff = 0.0 # FIXED: Starts at 0, baseflow added later
            
            total_inflow_vol = 0.0
            total_spill_vol = 0.0
            
            inflow_series_hourly = []
            outflow_series_hourly = []
            storage_series_hourly = []
            
            hours_above_crest = 0.0
            time_to_first_spill = None
            
            for hour, rain in enumerate(rain_mm_hr):
                target_runoff_m3s = (rain / 1000.0 / 3600.0) * CATCHMENT_AREA_M2 * runoff_coef
                
                hour_inflow_sum = 0
                hour_outflow_sum = 0
                
                for step in range(SUB_STEPS_PER_HOUR):
                    # 1. Linear Catchment Routing
                    dq = (target_runoff_m3s - catchment_runoff) / K_LAG_HOURS
                    catchment_runoff += dq * (DT_SECONDS / 3600.0)
                    inflow_m3s = catchment_runoff + BASEFLOW_M3S # FIXED: Baseflow added once
                    
                    # 2. Level-Pool Storage Approximation (Predictor)
                    storage_preliminary = storage + (inflow_m3s * DT_SECONDS)
                    
                    # 3. Spillway Outflow
                    if storage_preliminary > RESERVOIR_CAPACITY_M3:
                        head_m = (storage_preliminary - RESERVOIR_CAPACITY_M3) / SURFACE_AREA_M2
                        outflow_m3s = WEIR_COEF * SPILLWAY_LENGTH_M * (head_m ** 1.5)
                        hours_above_crest += (1.0 / SUB_STEPS_PER_HOUR)
                        
                        # FIXED: Safe check for hour 0.0 spill
                        if time_to_first_spill is None:
                            time_to_first_spill = hour + (step / SUB_STEPS_PER_HOUR)
                    else:
                        outflow_m3s = 0.0
                        
                    # 4. Final Mass Balance Update (Corrector)
                    storage += (inflow_m3s - outflow_m3s) * DT_SECONDS
                    storage = max(storage, 0.0)
                    
                    total_inflow_vol += (inflow_m3s * DT_SECONDS)
                    total_spill_vol += (outflow_m3s * DT_SECONDS)
                    
                    hour_inflow_sum += inflow_m3s
                    hour_outflow_sum += outflow_m3s
                    
                inflow_series_hourly.append(hour_inflow_sum / SUB_STEPS_PER_HOUR)
                outflow_series_hourly.append(hour_outflow_sum / SUB_STEPS_PER_HOUR)
                storage_series_hourly.append(storage)
            
            # Save baseline for plotting (90% full, C=0.50)
            if runoff_coef == 0.50 and initial_pct == 0.90: 
                storms_df[f'{T}yr_Inflow_m3s'] = inflow_series_hourly
                storms_df[f'{T}yr_Spill_m3s'] = outflow_series_hourly
                storms_df[f'{T}yr_Storage_m3'] = storage_series_hourly
            
            mass_balance_error = (INITIAL_STORAGE + total_inflow_vol) - (total_spill_vol + storage)
            
            results.append({
                'Init_Store': f"{int(initial_pct*100)}%",
                'C': runoff_coef,
                'T(yr)': T,
                'Peak Inflow(m3/s)': round(max(inflow_series_hourly), 2),
                'Peak Spill(m3/s)': round(max(outflow_series_hourly), 2),
                'Time to Spill(hr)': round(time_to_first_spill, 2) if time_to_first_spill is not None else "No Spill",
                'Spill Duration(hr)': round(hours_above_crest, 2),
                'Total Spill Vol(m3)': round(total_spill_vol, 2),
                'Mass Error(m3)': round(mass_balance_error, 2)
            })

results_df = pd.DataFrame(results)
results_df.to_csv('System_Dynamics/SD_Scenario_Sensitivity.csv', index=False)

print("\n" + "="*115)
print("SYSTEM DYNAMICS: CUMULATIVE SPILLWAY DISCHARGE SUMMARY")
print("="*115)
print(results_df.to_string(index=False))
print("="*115)