import os

log_batch_size = 50
uri = os.getenv("MONGODB_URI")
db_name = "search-logs"
search_requests_collection = "search-requests"
similar_courses_requests_collection = "similar-courses-requests"