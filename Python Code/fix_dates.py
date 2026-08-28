import pandas as pd

# Load the master dataset
df = pd.read_csv('Eleyele_HydroMet_Master.csv')

# If the first column is an unnamed index containing dates, name it 'Date'
if 'Unnamed: 0' in df.columns or df.columns[0] != 'Date':
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

# Ensure it is a proper datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Save it back clean
df.to_csv('Eleyele_HydroMet_Master.csv', index=False)
print("Master dataset dates successfully fixed and saved.")