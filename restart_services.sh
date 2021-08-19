#!/usr/bin/env bash

xhost +localhost \
&& docker-compose up --build --force-recreate

read -n 1 -s -r -p "Press any key to exit"
