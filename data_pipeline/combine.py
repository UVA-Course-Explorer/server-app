import os
import glob
import pandas as pd
import uuid

# Set the directory where your CSV files are located
directory = '.'

# List all CSV files in the directory
csv_files = glob.glob(os.path.join(directory, '*.csv'))

# Initialize an empty list to store the DataFrames
dfs = []

# Loop through each CSV file and append its contents to the dfs list
for file in csv_files:
    data = pd.read_csv(file)
    dfs.append(data)

# Concatenate all the DataFrames in the list into one
combined_data = pd.concat(dfs, ignore_index=True)

# Add a new column with unique IDs
combined_data['unique_id'] = [str(uuid.uuid4()) for _ in range(len(combined_data))]

# Save the combined data to a new CSV file
combined_data.to_csv('combined.csv', index=False)





