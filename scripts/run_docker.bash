#!/usr/bin/env bash

n_cores=$(grep -c ^processor /proc/cpuinfo)
avail_cores=$(($n_cores - 4))
if [ $avail_cores -lt 4 ]
then
    avail_cores=4
fi
cpus=$avail_cores
mem="24g"

# XServer
xsock="/tmp/.X11-unix"
xauth="/root/.Xauthority" 

docker run --rm -it \
    --user=root \
    --privileged \
    --env="DISPLAY"=$DISPLAY \
    --volume="$XAUTHORITY:$xauth:ro" \
    --env=XAUTHORITY=$xauth \
    --volume=$(pwd):/app/:rw \
    --volume=$xsock:$xsock:rw \
    --network=host \
    --cpus=$cpus \
    --memory=$mem \
    --gpus all \
    --ipc host \
    ravihammond/obl-project \
    ${@:-bash}

