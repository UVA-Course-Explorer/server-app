from flask import Flask, request
from search.semantic_search import SemanticSearch

app = Flask(__name__)

semantic_search = SemanticSearch()

@app.route('/helloWorld')  # Change the route to match your URL
def hello():
    return "Hello World!"

# Test route
@app.route('/members')
def get_members():
    return {'members': ['John', 'Paul', 'George', 'Ringo']}


@app.route('/search', methods=['POST'])
def search():
    search_input = request.json['searchInput']
    json_results = semantic_search.get_top_results_json(search_input, n=10)
    return json_results


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)  # Listen on 0.0.0.0:8080


# docker build --tag server-app .
# docker run -d -p 8080:8080 server-app
