import pymongo
import db_config
import time
import certifi
from motor.motor_asyncio import AsyncIOMotorClient

class SearchLogger:
    def __init__(self):
        self.search_requests = []
        self.similar_courses_requests = []

    async def insert_documents(self, docs, collection_name):
        client = None
        try:
            # Connect to MongoDB
            client = AsyncIOMotorClient(db_config.uri)
            # client = pymongo.MongoClient(db_config.uri, tlsCAFile=certifi.where())
            db = client[db_config.db_name]

            collection = db[collection_name]
            # Insert documents
            result = await collection.insert_many(docs)
            print(f"Saved {len(result.inserted_ids)} searches")
        except Exception as e:
            print(f"Error when saving requests to {collection_name} database: {e}")
        finally:
            # Close the connection
            if client:
                client.close()


    async def log_everything(self):
        if len(self.search_requests) > 0:
            await self.insert_documents(self.search_requests, db_config.search_requests_collection)
            self.search_requests.clear()

        if len(self.similar_courses_requests) > 0:
            self.insert_documents(self.similar_courses_requests, db_config.similar_courses_requests_collection)
            self.similar_courses_requests.clear()


    async def log_search_request(self, search_request):
        search_request['timestamp'] = time.time()
        self.search_requests.append(search_request)
        if len(self.search_requests) >= db_config.log_batch_size:
            await self.insert_documents(self.search_requests, db_config.search_requests_collection)
            self.search_requests.clear()


    async def log_similar_courses_request(self, similar_courses_request):
        similar_courses_request['timestamp'] = time.time()
        self.similar_courses_requests.append(similar_courses_request)
        if len(self.similar_courses_requests) >= db_config.log_batch_size:
            await self.insert_documents(self.similar_courses_requests, db_config.similar_courses_requests_collection)
            self.similar_courses_requests.clear()