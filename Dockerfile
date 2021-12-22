FROM python:3.6.1-alpine
WORKDIR /docker-flask-test
ADD . /docker-flask-test
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python","app.py"]