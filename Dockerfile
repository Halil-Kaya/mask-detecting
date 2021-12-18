FROM python:3.6.1-alpine
# define the present working directory
WORKDIR /docker-flask-test
# copy the contents into the working dir
ADD . /docker-flask-test
RUN pip install --upgrade pip
# run pip to install the dependencies of the flask app
RUN pip install -r requirements.txt
# define the command to start the container
CMD ["python","app.py"]