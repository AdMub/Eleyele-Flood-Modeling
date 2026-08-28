import geoglows
import pandas as pd

# The exact River ID for your segment of the Ona River
river_id = 140577794

# Retrieve the historical simulation data
# The ERA5 reanalysis dataset is used to produce a retrospective simulation on each river.
historical_df = geoglows.data.retrospective(river_id=river_id)

# Save it as a CSV file in your Hydrological folder
historical_df.to_csv("Ona_River_Historical_Streamflow.csv")

print("Download complete! Your file is saved.")