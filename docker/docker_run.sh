#!/usr/bin/env bash
container_name=continuous_rl

nvidia-docker stop ${container_name}
nvidia-docker rm ${container_name}
nvidia-docker run -it -d --net=host --ipc=host \
-v $PWD:/continuous_rl \
-w /continuous_rl --name ${container_name} continuous_rl:pytorch-1.1 bash
