FROM ubuntu:latest
COPY /web/* /app
WORKDIR /app
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python ./server.py
