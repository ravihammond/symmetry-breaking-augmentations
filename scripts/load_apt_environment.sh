#!/bin/bash

#apt update && xargs -a <(awk -F/ '{print $1}' /apt_environment_short.txt) apt install -y
apt update && xargs -a <(awk '{gsub("/"," ",$0); print $1"="$3}' /apt_environment.txt) apt install -y
#apt update && xargs -a /apt_environment_short.txt apt install -y
