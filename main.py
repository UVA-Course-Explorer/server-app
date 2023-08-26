from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
import os

from search.semantic_search import SemanticSearch

app = FastAPI()

# CORS
origins = [
    "https://uvacourses.netlify.app", 
    "localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_origin_regex='https://deploy-preview-(\d+)--uvacourses\.netlify\.app',
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
api_keys = [os.environ.get('SERVER_APP_API_KEY')]

api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header in api_keys:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

semantic_search = SemanticSearch()

@app.get("/helloWorld")
async def hello():
    return "Hello World!"

@app.get("/members")
async def get_members():
    return {'members': ['John', 'Paul', 'George', 'Ringo']}


@app.post('/search')
async def search(request: Request, api_key: str = Security(get_api_key)):
    search_request = await request.json()
    academic_level_filter = search_request['academicLevelFilter']
    semester_filter = search_request['semesterFilter']
    search_input = search_request['searchInput']

    json_results = semantic_search.get_search_results(search_input, academic_level_filter=academic_level_filter, semester_filter=semester_filter, n=10)
    encoded_results = jsonable_encoder(json_results)
    return JSONResponse(content=encoded_results)


@app.post('/similar_courses')
async def similar_courses(request: Request, api_key: str = Security(get_api_key)):
    search_request = await request.json()
    
    mnemonic, catalog_number = search_request['mnemonic'], str(search_request['catalog_number'])
    academic_level_filter = search_request['academicLevelFilter']
    semester_filter = search_request['semesterFilter']

    json_results = semantic_search.get_similar_course_results(mnemonic, catalog_number, academic_level_filter=academic_level_filter, semester_filter=semester_filter, n=10)

    encoded_results = jsonable_encoder(json_results)
    return JSONResponse(content=encoded_results)


# FastAPI Things
    # uvicorn main:app --host 0.0.0.0 --port 8080 --reload
    # --reload      -> if u wanna reload the server when u change something

# Docker Things
    # docker build --tag server-app .
    # docker run -d -p 8080:8080 server-app

# PromQL Commands
    # sum(increase(fly_app_http_responses_count{app="server-app", status="200"}[$time]))


# Curl with API Key
'''
curl -X POST "http://localhost:8080/search" \
-H "Content-Type: application/json" \
-H "X-API-Key: <PUT API KEY HERE>" \
-d '{
    "academicLevelFilter": "all",
    "semesterFilter": "all",
    "searchInput": "machine learning"
}'
'''