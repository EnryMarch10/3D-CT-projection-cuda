#!/bin/bash

rm -f logs/{error,output}.txt
mkdir -p logs
nohup ./scripts/tests.sh < /dev/null > logs/output.txt 2> logs/error.txt &
