FROM ubuntu:22.10

#ARG cuda_devices
#ENV CUDA_VISIBLE_DEVICES=$cuda_devices 

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install wget git\
    cmake make vim sudo xgboost -y
RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip -qq
RUN pip3 install -U s3fs fsspec
RUN pip3 install ipdb shyaml

RUN useradd -rm -d /home/user -s /bin/bash -g root -G sudo -u 1000 user
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN chmod -R a+rwx /home/user

USER root

RUN git clone https://github.com/nmslib/nmslib.git .
RUN cd nmslib
COPY ptach_file.patch .
RUN git apply ptach_file.patch
RUN cd python_bindings
RUN pip3 install .
RUN cd ..
RUN rm -rf nmslib

COPY requirements.txt /tmp
RUN pip3 install -r /tmp/requirements.txt 
RUN pip3 install --no-deps macest

COPY configs/.vimrc /root/.vimrc
RUN git clone https://github.com/VundleVim/Vundle.vim.git /root/.vim/bundle/Vundle.vim

#USER user

