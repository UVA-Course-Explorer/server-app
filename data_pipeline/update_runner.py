import csv
from data_fetcher import DataFetcher
import aiohttp
import asyncio
import os
import json
import requests
import pandas as pd
from asyncio import Lock
import uuid
from data_pipeline import SearchDataGenerationPipeline


semesters_to_udpate = [1251, 1252, 1256, 1258]

previous_semester_file = 'combined.csv'
intermediate_csv_file = 'combined_mod.csv'
output_dir = "data_pipeline_output/"
replace_combined = False    # set to True if you want to replace the combined.csv file after population is complete

latest_semester = semesters_to_udpate[-1]
semaphore = None
data_save_lock = None
new_df = None

# remove_from_master
def filter_csv(input_file_name, output_file_name, semesters_to_udpate, column_name='strm'):
    with open(input_file_name, mode='r', newline='') as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames
        
        # Ensure the column to check exists in the CSV
        if column_name not in fieldnames:
            print(f"Error: The column '{column_name}' was not found in the input file.")
            return
        
        with open(output_file_name, mode='w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                if row[column_name] not in semesters_to_udpate:
                    writer.writerow(row)


async def fetch_data(session, url, group_key):
    print("Fetching data for group: ", group_key)
    async with semaphore:
        async with session.get(url) as response:
            return await response.json()


# Your main asynchronous function
async def process_data(group_key, group_data):
    strm = group_data['strm'].iloc[0]
    class_nbr = group_data['class_nbr'].iloc[0]
    print("Processing group: ", group_key, group_data['strm'].iloc[0], group_data['catalog_nbr'].iloc[0], group_data['subject'].iloc[0])
    url = f"https://sisuva.admin.virginia.edu/psc/ihprd/UVSS/SA/s/WEBLIB_HCX_CM.H_CLASS_SEARCH.FieldFormula.IScript_ClassDetails?institution=UVA01&term={strm}&class_nbr={class_nbr}"

    async with aiohttp.ClientSession() as session:
        api_response = await fetch_data(session, url, group_key)
        group_data['description'] = api_response['section_info']['catalog_descr']['crse_catalog_description']
        print("finished ", group_key)
        await save_data(group_data)


async def save_data(group_data):
    global new_df
    with await data_save_lock:
        new_df = pd.concat([new_df, group_data], ignore_index=True)  # Concatenate the rows


def main():
    global semaphore, data_save_lock, new_df

    ########## remove old semester code ---> generates intermediate_csv_file ##########
    # remove rows corresponding to semesters to update

    print("Filtering CSV for semesters to update")
    filter_csv(previous_semester_file, intermediate_csv_file, semesters_to_udpate, 'strm')
    print("CSV Filtered")

    ########## data_fetcher.py code ---> fetches data for specified semesters and generates csvs##########

    for strm in semesters_to_udpate:
        # fetch data for the specified semesters
        data_fetcher = DataFetcher(strm)
        data_fetcher.run()
        # asyncio.get_event_loop().wait()
        # asyncio.get_event_loop().close()

    ########## update_csv.py code ---> add descriptions to each of the csv's generated in the previous step ##########

    print("Updating CSVs")
    for strm in semesters_to_udpate:

        new_df = pd.DataFrame()
        print("Updating CSV for semester: ", strm)
        semaphore = asyncio.Semaphore(10)
        data_save_lock = Lock()

        # Load your data into a Pandas DataFrame
        df = pd.read_csv(f'{strm}.csv')
        df = df[df['strm'] == strm]

        # Group by unique combinations of columns A, B, and C
        unique_combinations = df.groupby(['strm', 'catalog_nbr', 'subject'])

        async def main_updater():
            tasks = []
            for group_key, group_data in unique_combinations:
                task = process_data(group_key, group_data)
                tasks.append(task)

            await asyncio.gather(*tasks)

        loop = asyncio.get_event_loop()

        try:
            loop.run_until_complete(main_updater())
        
        except Exception as e:
            print(f"Error updating CSV for semester {strm}: {e}")
        
        finally:
            # loop.close()
            print("closed loop")

        print(new_df.head())
        for i in range(100):
            print("SAVING UPDATED CSV!!!")
        # save the updated data to a new CSV file
        new_df.to_csv(f'{strm}.csv', mode='w', header=True, index=False)
    
    print("CSVs Updated")

    ################### combine.py the data in the individual generated csv's with the new code ########################

    print("Combining CSVs")

    # combine the data in intermediate_csv_file with new data
    old_df = pd.read_csv(intermediate_csv_file)

    # Initialize an empty list to store the DataFrames
    dfs = []

    for strm in semesters_to_udpate:
        data = pd.read_csv(f'{strm}.csv')
        dfs.append(data)
    
    # Concatenate all the DataFrames in the list into one
    combined_data = pd.concat(dfs, ignore_index=True)

    # Add a new column with unique IDs
    combined_data['unique_id'] = [str(uuid.uuid4()) for _ in range(len(combined_data))]

    # combine with old data
    combined_data = pd.concat([old_df, combined_data], ignore_index=True)

    # Save the combined data to a new CSV file (same name as intermediate_csv_file)
    combined_data.to_csv(intermediate_csv_file, index=False)

    if replace_combined:
        os.remove(previous_semester_file)
        os.rename(intermediate_csv_file, previous_semester_file)
    
    print("Master CSV Generated")

    ########## data_pipeline.py code ---> runs the data pipeline on the combined csv file ##########
    # run data pipeline

    print("Running Data Pipeline")
    df = pd.read_csv(intermediate_csv_file)

    pipeline = SearchDataGenerationPipeline(output_dir, latest_semester)
    pipeline.run(df, output_dir, latest_semester)

    print("Data Pipeline Run")


if __name__ == '__main__':
    main()








    










# run data_fetcher for the specified semesters

# run update_csv for the specified semesters

# combine with main csv file

# run data pipeline

# push <3