FROM python:3.11.4

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "uvicorn", "main:app" , "--host", "0.0.0.0", "--port", "8080"]  # Specify the port as well
