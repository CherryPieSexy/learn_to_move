#!/usr/bin/env bash
cd ..
docker build -t continuous_rl:pytorch-1.1 -f docker/Dockerfile .
