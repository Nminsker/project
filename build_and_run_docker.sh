#!/bin/bash

cuda="${@:1:1}"
args="${@:2:3}"

docker build -t "omer:v1" -f ./Dockerfile .\
    --build-arg cuda_devices="$cuda"

docker run --ipc=host --env=DISPLAY -p 8988:8988 \
	       	--rm -it -v `pwd`:`pwd` -w `pwd` omer:v1 $args
