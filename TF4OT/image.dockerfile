FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

RUN set -xe
RUN apt-get -yqq update
RUN apt-get -yqq install python3-pip
RUN pip3 install --upgrade pip
RUN apt-get -yqq install python3-venv
RUN apt-get -yqq install rsync
RUN apt-get -y install git
RUN apt-get install nvtop

# build from the repository root: docker build -f TF4OT/image.dockerfile .
COPY ./TF4OT/requirements.txt /tmp/requirements.txt
#ENV PATH="/home/software/bin:${PATH}"
#ENV STAGING_DIR=/staging/jaenmarquez

RUN pip3 install -r /tmp/requirements.txt