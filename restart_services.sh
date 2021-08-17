#!/usr/bin/env bash

docker-compose up --build --force-recreate

read -n 1 -s -r -p "Press any key to exit"
