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
    search_request = await request.json()
    academic_level_filter = search_request['academicLevelFilter']
    semester_filter = search_request['semesterFilter']
    search_input = search_request['searchInput']
    return_graph_data = search_request['getGraphData']

    json_results = semantic_search.get_search_results(search_input, academic_level_filter=academic_level_filter, semester_filter=semester_filter, n=10, return_graph_data=return_graph_data)
    encoded_results = jsonable_encoder(json_results)
    return JSONResponse(content=encoded_results)



@app.post('/similar_courses')
async def similar_courses(request: Request):
    search_request = await request.json()
    
    mnemonic, catalog_number = search_request['mnemonic'], str(search_request['catalog_number'])
    academic_level_filter = search_request['academicLevelFilter']
    semester_filter = search_request['semesterFilter']
    return_graph_data = search_request['getGraphData']

    json_results = semantic_search.get_similar_course_results(mnemonic, catalog_number, academic_level_filter=academic_level_filter, semester_filter=semester_filter, n=10, return_graph_data=return_graph_data)

    encoded_results = jsonable_encoder(json_results)
    return JSONResponse(content=encoded_results)


# FastAPI Things
    # uvicorn main:app --host 0.0.0.0 --port 8080 --reload
    # --reload      -> if u wanna reload the server when u change something

# Docker Things
    # docker build --tag server-app .
    # docker run -d -p 8080:8080 server-app