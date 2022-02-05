#n_cores=$(grep -c ^processor /proc/cpuinfo)
#avail_cores=$(($n_cores - 4))
#if [ $avail_cores -lt 4 ]
#then
    #avail_cores=4
#fi
#cpus=$avail_cores
#mem="24g"

## XServer
#xsock="/tmp/.X11-unix"
#xauth="/tmp/.docker.xauth"

#docker run -it \
    #--user=$(id -u ${USER}):$(id -g ${USER}) \
    #--volume=${PWD}:/home/${USER}:rw \
    #--workdir=/home/${USER}/src \
    #--env="DISPLAY"=$DISPLAY \
    #--volume="/etc/group:/etc/group:ro"   \
    #--volume="/etc/passwd:/etc/passwd:ro" \
    #--volume="/etc/shadow:/etc/shadow:ro" \
    #--volume=$xsock:$xsock:rw \
    #--volume=$xauth:$xauth:rw \
    #--env=XAUTHORITY=$xauth \
    #--network=host \
    #--cpus=$cpus \
    #--memory=$mem \
    #--gpus all \
    #cuda-conda

#!/usr/bin/env bash

docker run --rm -it \
    --volume=$(pwd):/app/:rw \
    --gpus all \
    --ipc host \
    ravihammond/obl-project \
    ${@:-bash}
