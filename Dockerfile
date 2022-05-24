FROM ubuntu
WORKDIR /docker-flask-test
ADD . /docker-flask-test
RUN apt update
RUN apt upgrade -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y gstreamer1.0-plugins-bad
RUN apt-get update
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN pip install --upgrade pip
RUN pip install opencv-python
RUN pip install -r requirements.txt
RUN pip install Pillow
CMD ["python3","app.py"]