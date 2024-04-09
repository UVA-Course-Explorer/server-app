import asyncio
import aiohttp
import sqlite3
import pandas as pd
# import pickle

# currently need to make sure num_pages_in_batch is greater than number of pages in SIS, cuz SIS rate limits after one iteration leading to a request timeout
# there's probably a better way to get around this

class DataFetcher:
    def __init__(self, path_to_db, table_name, strm, num_pages_in_batch=100, save_to_db=True):
        self.path_to_db = path_to_db
        self.table_name = table_name
        self.strm = strm
        self.num_pages_in_batch = num_pages_in_batch
        self.courses = []
        self.save_to_db = save_to_db


    def get_base_url(self):
        return f"https://sisuva.admin.virginia.edu/psc/ihprd/UVSS/SA/s/WEBLIB_HCX_CM.H_CLASS_SEARCH.FieldFormula.IScript_ClassSearch?institution=UVA01&term={self.strm}"


    async def fetch_courses(self, session, page):
        url = self.get_base_url() + f"&page={page}"
        print(f"Fetching data for page {page}")
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                print(f"Got results for page {page} of {self.strm}")
                return data
            else:
                print(f"Failed to fetch data for page {page}")
                return []


    async def get_all_courses_in_semester(self):
        async with aiohttp.ClientSession() as session:
            in_progress = True
            iteration = 0
            while in_progress:
                start_page = 1 + iteration * self.num_pages_in_batch
                end_page = self.num_pages_in_batch + start_page
                print(f"Fetching pages {start_page} to {end_page}")
                tasks = [asyncio.create_task(self.fetch_courses(session, page)) for page in range(start_page, end_page)]
                results = await asyncio.gather(*tasks)  # Fetch all pages concurrently
                for page, data in enumerate(results):
                    if data == []:
                        print(f"Page {page} had no results, so we're done")
                        in_progress = False
                    for course in data:
                        self.courses.append(course)
                iteration += 1
            
            # with open("sis_data.pkl", "wb") as f:
            #     pickle.dump(self.courses, f)
            print("Done fetching data from SIS")
    

    def run(self):
        # fetch data from SIS
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.get_all_courses_in_semester())
        loop.close()

        # with open("sis_data.pkl", "rb") as f:
        #     self.courses = pickle.load(f)
        
        df = pd.DataFrame(self.courses)
        df.to_csv(f"{self.strm}.csv", mode="w", header=True, index=False)

        if self.save_to_db:
            df = pd.read_csv("output.csv")
            conn = sqlite3.connect(self.path_to_db)
            df.to_sql(self.table_name, conn, if_exists='replace', index=False)





data_fetcher = DataFetcher("data_1248.db", "sessions", 1248, save_to_db=False)
data_fetcher.run()