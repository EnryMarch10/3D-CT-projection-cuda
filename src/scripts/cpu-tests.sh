#!/bin/bash

if [ -z "$DIR_RESULTS" ]; then
  echo "$0> The variable 'DIR_RESULTS' is unset." >&2
  exit 1
fi

if [ ! -d "$DIR_RESULTS" ]; then
  echo "$0> The variable 'DIR_RESULTS=$DIR_RESULTS' is not a valid directory." >&2
  exit 1
fi

if [ $# -gt 1 ]; then
  echo "$0> Usage: $0 [N_REPS]" >&2
  exit 1
fi

N_REPS=10

if [ $# -eq 1 ]; then
  if [[ "$1" =~ ^-?[0-9]+$ ]]; then
    N_REPS="$1"
  else
    echo "$0> N_REPS has to be an integer number" >&2
    exit 1
  fi
fi

shopt -s nullglob

for SOURCE in omp-*.c; do
  if [[ $SOURCE != *-test.c ]]; then
    echo "# $0> Testing $SOURCE at `date "+%Y-%m-%d %I:%M:%S %p"`"
    NAME="$( basename -s .c "$SOURCE" )"

    echo -n "TIME: " > "$DIR_RESULTS/tput-$NAME.txt"
    date "+%Y-%m-%d %I:%M:%S %p" >> "$DIR_RESULTS/tput-$NAME.txt"
    echo "SOURCE: $SOURCE" >> "$DIR_RESULTS/tput-$NAME.txt"
    echo >> "$DIR_RESULTS/tput-$NAME.txt"
    mkdir -p ./inputs
    mkdir -p "./outputs/tput/$NAME"
    INPUT_MAX_DIM=1500 PROG_INPUT=./build/bin/inputgen DIR_INPUT=./inputs DIR_OUTPUT="./outputs/tput/$NAME" "$(dirname "$0")/demo-tput-openmp.sh" "./build/bin/$NAME" "$N_REPS" >> "$DIR_RESULTS/tput-$NAME.txt"
  fi
done

shopt -u nullglob
