FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
      && apt-get install --no-install-recommends --no-install-suggests -y gnupg2 ca-certificates \
            git build-essential libopencv-dev \
      && rm -rf /var/lib/apt/lists/*

RUN cd /root && git clone https://github.com/AlexeyAB/darknet.git && cd darknet && make

COPY files/obj.names /root/darknet/data/obj.names 
COPY files/obj.data /root/darknet/data/obj.data
COPY files/yolov4-obj.cfg /root/darknet/cfg/yolov4-obj.cfg
COPY files/yolov4-obj_last.weights /root/darknet/yolov4-obj_last.weights
COPY files/classes.txt /root/darknet/classes.txt
COPY files/schemes_detector.py /root/darknet/schemes_detector.py

WORKDIR /root/darknet
ADD files /files

CMD python /root/darknet/schemes_detector.py
