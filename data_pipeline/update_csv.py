import csv
import aiohttp
import asyncio
import os
import json
import requests
import pandas as pd
from asyncio import Lock


semaphore = asyncio.Semaphore(10)  
data_save_lock = Lock()

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


<<<<<<< Updated upstream

#  1241, 1242

strm = 1248

=======
strm = 1251
>>>>>>> Stashed changes

# Load your data into a Pandas DataFrame
df = pd.read_csv(f'{strm}.csv')


df = df[df['strm'] == strm]


# Group by unique combinations of columns A, B, and C
unique_combinations = df.groupby(['strm', 'catalog_nbr', 'subject'])

new_df = pd.DataFrame()

# Create an event loop and run the asynchronous tasks
async def main():
    tasks = []
    for group_key, group_data in unique_combinations:
        task = process_data(group_key, group_data)
        tasks.append(task)

    await asyncio.gather(*tasks)

# Run the event loop to start the asynchronous tasks
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

# Save the final modified DataFrame to a new spreadsheet
new_df.to_csv(f'updated_{strm}.csv', index=False)