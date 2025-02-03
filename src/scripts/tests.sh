#!/bin/bash

echo "$0> make purge"
make purge
if [ $? -ne 0 ]; then
  echo "$0> Command 'make purge' failed. Aborting script." >&2
  exit 1
fi

echo "$0> make inputgen"
make inputgen
if [ $? -ne 0 ]; then
  echo "$0> Command 'make inputgen' failed. Aborting script." >&2
  exit 1
fi

echo "$0> make omp"
make omp
if [ $? -ne 0 ]; then
  echo "$0> Command 'make omp' failed. Aborting script." >&2
  exit 1
fi

echo "$0> make cuda"
make cuda
if [ $? -ne 0 ]; then
  echo "$0> Command 'make cuda' failed. Aborting script." >&2
  exit 1
fi

RESULTS_CPU=./results/cpu
RESULTS_GPU=./results/gpu

mkdir -p "$RESULTS_CPU"
mkdir -p "$RESULTS_GPU"
echo "$0> ### Start CPU tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
echo "Working..."
DIR_RESULTS="$RESULTS_CPU" "$(dirname "$0")/cpu-tests.sh"
if [ $? -ne 0 ]; then
  echo "$0> ### CPU tests failed! Aborting script ###" >&2
  exit 1
fi
echo "$0> ### Ended CPU tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
echo "$0> ### Start GPU tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
echo "Working..."
DIR_RESULTS="$RESULTS_GPU" "$(dirname "$0")/gpu-tests.sh"
if [ $? -ne 0 ]; then
  echo "$0> ### GPU tests failed! Aborting script ###" >&2
  exit 1
fi
echo "$0> ### Ended GPU tests at `date "+%Y-%m-%d %I:%M:%S %p"` ###"
