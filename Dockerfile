FROM ubuntu
WORKDIR /docker-flask-test
ADD . /docker-flask-test
RUN apt update
RUN apt upgrade -y
RUN apt install wget -y
RUN apt install curl -y
RUN apt-get install software-properties-common -y
RUN curl -L -O "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb" \ && dpkg -i cuda-keyring_1.0-1_all.deb \
RUN apt-get install gnupg -y
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y nvidia-kernel-source-460
RUN apt-get -y install cuda
RUN apt-get install -y gstreamer1.0-plugins-bad
RUN apt install nvidia-cuda-toolkit -y
#RUN add-apt-repository ppa:mc3man/gstffmpeg-keep
#RUN apt-get install gstreamer0.10-ffmpeg -y
RUN apt-get update
RUN apt-get update
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN pip install --upgrade pip
RUN pip install flask_socketio
RUN pip install opencv-python
RUN pip install -r requirements.txt
RUN pip install Pillow
CMD ["python3","app.py"]