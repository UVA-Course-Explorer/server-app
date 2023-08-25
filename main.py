from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

from search.semantic_search import SemanticSearch

app = FastAPI()

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

semantic_search = SemanticSearch()

@app.get("/helloWorld")
async def hello():
    return "Hello World!"

@app.get("/members")
async def get_members():
    return {'members': ['John', 'Paul', 'George', 'Ringo']}


@app.post('/search')
async def search(request: Request):
    search_input = await request.json()
    json_results = semantic_search.get_top_results_json(search_input['searchInput'], n=10)
    encoded_results = jsonable_encoder(json_results)
    return JSONResponse(content=encoded_results)

# FastAPI Things
    # uvicorn main:app --host 0.0.0.0 --port 8080 --reload
    # --reload      -> if u wanna reload the server when u change something

# Docker Things
    # docker build --tag server-app .
    # docker run -d -p 8080:8080 server-app