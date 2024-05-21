import pandas as pd
import os
import uuid  # Add this import statement

# Specify the files you want to combine
files_to_combine = ['updated_1246.csv', 'updated_1248.csv', 'combined_mod.csv']  # Add more files as needed

# Initialize an empty list to store the DataFrames
dfs = []

# Loop through each file and append its contents to the dfs list
for file in files_to_combine:
    if os.path.exists(file):  # Check if the file exists
        data = pd.read_csv(file)
        dfs.append(data)
    else:
        print(f"File '{file}' not found.")

# Check if there are any DataFrames to combine
if len(dfs) > 0:
    # Concatenate all the DataFrames in the list into one
    combined_data = pd.concat(dfs, ignore_index=True)

    # Add a new column with unique IDs
    combined_data['unique_id'] = [str(uuid.uuid4()) for _ in range(len(combined_data))]

    # Save the combined data to a new CSV file
    combined_data.to_csv('combined.csv', index=False)
    print("Combined CSV file created successfully.")
else:
    print("No files to combine.")
