FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
      && apt-get install --no-install-recommends --no-install-suggests -y gnupg2 ca-certificates \
            git build-essential libopencv-dev \
      && apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils \
      && rm -rf /var/lib/apt/lists/* 

RUN cd /root && git clone https://github.com/AlexeyAB/darknet.git && cd darknet && make


COPY req.txt /req.txt
RUN python3.8 -m pip install --no-cache-dir -r /req.txt

# Add user
RUN adduser --quiet --disabled-password qtuser && usermod -a -G audio qtuser

# This fix: libGL error: No matching fbConfigs or visuals found
ENV LIBGL_ALWAYS_INDIRECT=1

# Install Python 3, PyQt5
RUN apt-get update && apt-get install -y python3-pyqt5

COPY files/obj.names /root/darknet/data/obj.names 
COPY files/obj.data /root/darknet/data/obj.data
COPY files/yolov4-obj.cfg /root/darknet/cfg/yolov4-obj.cfg
COPY files/yolov4-obj_last.weights /root/darknet/yolov4-obj_last.weights
COPY files/classes.txt /root/darknet/classes.txt
COPY files/schemes_detector.py /root/schemes_detector.py


WORKDIR /root/darknet


CMD ["python3", "/root/schemes_detector.py"]