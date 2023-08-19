from flask import Flask

app = Flask(__name__)

@app.route('/helloWorld')  # Change the route to match your URL
def hello():
    return "Hello World!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)  # Listen on 0.0.0.0:8080


# docker build --tag server-app .
# docker run -d -p 5000:5000 server-app
