FROM python:3.8.3-buster

WORKDIR /usr/src/app

COPY ./requirements.txt ./
RUN pip3 install -r requirements.txt

EXPOSE 5000
CMD [ "python3", "app.py" ]
