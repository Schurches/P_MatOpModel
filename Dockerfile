FROM ubuntu:latest
#FROM python:3.6.7-alpine3.6
COPY /web/* /app/
WORKDIR /app
#VOLUME ./app /app/
#RUN apt-get update -y
#RUN apt-get install -y python-pip python3 python3-dev build-essential
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && apt-get install nano \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
RUN pip3 install -r /app/requirements.txt
EXPOSE 5000
CMD python ./server.py
