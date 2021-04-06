#!/bin/bash

#source ./set_env.sh

port=$(shuf -i8000-9999 -n1)
ip=$(hostname -i)
node=$(hostname -a)

Xvfb :0 -screen 0 800x600x16&

echo host=${node} ip=${ip} port=${port}
jupyter lab --no-browser --port=${port} --ip=${ip}
