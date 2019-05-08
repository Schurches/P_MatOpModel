FROM ubuntu:latest
LABEL maintainer="Zach Bloomquist <zach@bloomqu.ist>"
RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-pip nginx
RUN pip3 install uwsgi
RUN apt-get install nano
COPY ./web/ ./app
WORKDIR ./app
RUN pip3 install -r /app/requirements.txt
COPY ./nginx.conf /etc/nginx/sites-enabled/default
CMD service nginx start && uwsgi -s /tmp/uwsgi.sock --chmod-socket=666 --manage-script-name --mount /=app:app
