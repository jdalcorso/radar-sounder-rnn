#!/bin/bash
# $1: image name
# $2: tag name
docker build --build-arg="USERID=$(id -u)" \
    --build-arg="GROUPID=$(id -g)" \
    --build-arg="REPO_DIR=$(pwd | sed "s/$USER/containeruser/")" \
    -t $USER/$1:$2 .
docker run -it -h $1 --name $1_$USER \
    -u $(id -u):$(id -g) \
    -v /home/$USER:/home/containeruser \
    -v /media:/media \
    -w /home/containeruser \
    $USER/$1:$2
