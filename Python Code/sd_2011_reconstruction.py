import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('System_Dynamics', exist_ok=True)

print("--- Step 1: Reconstructing the August 2011 Ibadan Storm ---")
# Source: World Bank Post-Disaster Assessment
# Total Rainfall: ~187.5 mm.
# Extreme Burst: ~140.63 mm concentrated in the central hour of the storm (approximating the historical 70-min burst).

hours = np.arange(1, 6) # 5 hour storm
rain_2011_hourly = np.zeros(5)

# Alternating block placing the massive 140.63mm burst at the peak (Hour 3)
# The remaining 46.87mm is distributed around it to total 187.5mm
rain_2011_hourly[0] = 10.0
rain_2011_hourly[1] = 20.0
rain_2011_hourly[2] = 140.63 # The catastrophic burst
rain_2011_hourly[3] = 11.87
rain_2011_hourly[4] = 5.0

# =====================================================================
# --- Step 2: System Dynamics Routing ---
# =====================================================================
print("--- Step 2: Routing 2011 Storm through Eleyele Reservoir ---")

CATCHMENT_AREA_M2 = 325_000_000  
# World Bank Estimates:
SPILLWAY_LENGTH_M = 98.0         
WEIR_COEF = 2.0                  
RESERVOIR_CAPACITY_M3 = 7_040_000 # Provisional capacity
SURFACE_AREA_M2 = 1_500_000      
BASEFLOW_M3S = 2.0

# Worst-case scenario assumptions for 2011 disaster:
RUNOFF_COEF = 0.60 # High runoff assumed for extreme wet-season saturation
INITIAL_STORAGE = RESERVOIR_CAPACITY_M3 * 0.90 # 90% full antecedent condition
K_LAG_HOURS = 9.96 # Kirpich calculation

SUB_STEPS_PER_HOUR = 12
DT_SECONDS = 3600.0 / SUB_STEPS_PER_HOUR
DECAY = np.exp(- (1.0 / SUB_STEPS_PER_HOUR) / K_LAG_HOURS)

storage = INITIAL_STORAGE
catchment_runoff = 0.0 
cumulative_spill_volume = 0.0

inflow_2011 = []
outflow_2011 = []
storage_2011 = []

hours_above_crest = 0.0
time_to_first_spill = None

for hour_idx, rain in enumerate(rain_2011_hourly):
    target_runoff_m3s = (rain / 1000.0 / 3600.0) * CATCHMENT_AREA_M2 * RUNOFF_COEF
    
    hour_inflow_sum = 0
    hour_outflow_sum = 0
    
    for step in range(SUB_STEPS_PER_HOUR):
        # Exact stable exponential update for catchment routing
        catchment_runoff = target_runoff_m3s + (catchment_runoff - target_runoff_m3s) * DECAY
        inflow_m3s = catchment_runoff + BASEFLOW_M3S 
        
        storage_preliminary = storage + (inflow_m3s * DT_SECONDS)
        
        if storage_preliminary > RESERVOIR_CAPACITY_M3:
            head_m = (storage_preliminary - RESERVOIR_CAPACITY_M3) / SURFACE_AREA_M2
            outflow_m3s = WEIR_COEF * SPILLWAY_LENGTH_M * (head_m ** 1.5)
            hours_above_crest += (1.0 / SUB_STEPS_PER_HOUR)
            
            if time_to_first_spill is None:
                time_to_first_spill = hour_idx + (step / SUB_STEPS_PER_HOUR)
        else:
            outflow_m3s = 0.0
            
        storage += (inflow_m3s - outflow_m3s) * DT_SECONDS
        storage = max(storage, 0.0)
        cumulative_spill_volume += (outflow_m3s * DT_SECONDS)
        
        hour_inflow_sum += inflow_m3s
        hour_outflow_sum += outflow_m3s
        
    inflow_2011.append(hour_inflow_sum / SUB_STEPS_PER_HOUR)
    outflow_2011.append(hour_outflow_sum / SUB_STEPS_PER_HOUR)
    storage_2011.append(storage)

# Run the simulation for an extra 10 hours to watch the recession limb
for hour_idx in range(5, 15):
    hour_inflow_sum = 0
    hour_outflow_sum = 0
    target_runoff_m3s = 0.0 # Rain has stopped
    
    for step in range(SUB_STEPS_PER_HOUR):
        catchment_runoff = target_runoff_m3s + (catchment_runoff - target_runoff_m3s) * DECAY
        inflow_m3s = catchment_runoff + BASEFLOW_M3S 
        
        storage_preliminary = storage + (inflow_m3s * DT_SECONDS)
        
        if storage_preliminary > RESERVOIR_CAPACITY_M3:
            head_m = (storage_preliminary - RESERVOIR_CAPACITY_M3) / SURFACE_AREA_M2
            outflow_m3s = WEIR_COEF * SPILLWAY_LENGTH_M * (head_m ** 1.5)
            hours_above_crest += (1.0 / SUB_STEPS_PER_HOUR)
        else:
            outflow_m3s = 0.0
            
        storage += (inflow_m3s - outflow_m3s) * DT_SECONDS
        storage = max(storage, 0.0)
        cumulative_spill_volume += (outflow_m3s * DT_SECONDS)
        
        hour_inflow_sum += inflow_m3s
        hour_outflow_sum += outflow_m3s
        
    inflow_2011.append(hour_inflow_sum / SUB_STEPS_PER_HOUR)
    outflow_2011.append(hour_outflow_sum / SUB_STEPS_PER_HOUR)
    storage_2011.append(storage)

print("\n" + "="*65)
print("EVENT RECONSTRUCTION: AUGUST 26, 2011 IBADAN FLOOD")
print("="*65)
print(f"Total Rainfall            : 187.5 mm")
print(f"Peak Simulated Inflow     : {max(inflow_2011):.2f} m³/s")
print(f"Peak Spillway Discharge   : {max(outflow_2011):.2f} m³/s")
print(f"Time to Spillway Release  : Hour {time_to_first_spill:.2f} of storm")
print(f"Duration of Spill Release : {hours_above_crest:.2f} hours")
print(f"Cumulative Overflow Volume: {cumulative_spill_volume / 1e6:.2f} Million m³")
print("="*65)

# =====================================================================
# --- Step 3: Visualization ---
# =====================================================================
print("\n--- Step 3: Generating Reconstruction Plot ---")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(10, 6))

total_hours = np.arange(1, 16)
ax1.plot(total_hours, inflow_2011, label='Simulated Catchment Inflow', color='dodgerblue', linewidth=2.5)
ax1.plot(total_hours, outflow_2011, label='Simulated Spillway Discharge', color='crimson', linewidth=2.5, linestyle='--')

# FIXED: Label changed to match the Y-axis units (Flow Rate)
ax1.fill_between(total_hours, 0, outflow_2011, color='crimson', alpha=0.2, label='Spillway Discharge Envelope')

ax1.set_title('SD Model Event Reconstruction: August 26, 2011 Ibadan Flood', fontsize=15, fontweight='bold')
ax1.set_xlabel('Time Since Storm Onset (Hours)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Flow Rate (m³/s)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
plt.xticks(np.arange(1, 16, 1))

plt.tight_layout()
plt.savefig('System_Dynamics/SD_2011_Reconstruction.png', dpi=300)
plt.close()
print("Saved Plot: System_Dynamics/SD_2011_Reconstruction.png")