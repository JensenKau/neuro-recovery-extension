FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN apt-get update -y 
RUN apt-get -y install curl
RUN apt-get install make
RUN apt-get -y install git

RUN apt-get install -y python3 python3-venv python3-pip

CMD ["/bin/bash"]