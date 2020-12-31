FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# install python 3
RUN apt-get update \
    && apt-get install -y python3-pip python3-dev \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip

# Latest Cmake
RUN apt-get install -y git apt-transport-https ca-certificates gnupg software-properties-common wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN apt-get update
RUN apt-get install -y cmake

# TODO: Install numpy with mkl

# Install known dependencies
RUN pip install torch torchvision
# ray doesn't install pandas, opencv but requires it
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip install pandas requests
# must be after pandas
# Ray from a specific commit
#RUN pip install https://ray-wheels.s3-us-west-2.amazonaws.com/master/d6f78f58dc7548217b46565a3dd42cd2e0133e66/ray-0.9.0.dev0-cp36-cp36m-manylinux1_x86_x64.whl
RUN pip install ray
# will install from wheel magically
RUN pip install ray[tune]
# must be on a separate line
RUN pip install ray[rllib]

# MINERL stuff
RUN add-apt-repository ppa:openjdk-r/ppa
RUN apt-get update
RUN apt-get install openjdk-8-jdk
RUN pip install --upgrade minerl

# Allow render from container
RUN apt-get install -y libsdl2-2.0
#ENV SDL_AUDIODRIVER "dummy"

WORKDIR /home

# Install requirements
COPY ./requirements.txt /home/MineRL/requirements.txt
RUN pip install -r /home/MineRL/requirements.txt

# Copy source and install with develop
COPY ./ /home/MineRL
WORKDIR /home/MineRL
RUN python setup.py develop

# script entry
ENTRYPOINT python /home/ray/python/ray/setup-dev.py --yes && /bin/bash