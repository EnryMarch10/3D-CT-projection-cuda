#!/bin/bash

if [ -z "$DIR_RESULTS" ]; then
  echo "$0> The variable 'DIR_RESULTS' is unset." >&2
  exit 1
fi

if [ ! -d "$DIR_RESULTS" ]; then
  echo "$0> The variable 'DIR_RESULTS=$DIR_RESULTS' is not a valid directory." >&2
  exit 1
fi

if [ -z "$INPUT_MAX_DIM" ]; then
  echo "$0> The variable 'INPUT_MAX_DIM' is unset." >&2
  exit 1
fi

if [[ ! "$INPUT_MAX_DIM" =~ ^-?[0-9]+$ ]]; then
  echo "$0> The variable 'INPUT_MAX_DIM' should be a valid integer" >&2
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

for SOURCE in cuda-*.cu; do
  if [[ $SOURCE != *-test.cu ]]; then
    echo "# $0> Testing $SOURCE at `date "+%Y-%m-%d %I:%M:%S %p"`"
    NAME="$( basename -s .cu "$SOURCE" )"

    RESULT="$DIR_RESULTS/tput-$NAME-$INPUT_MAX_DIM.txt"
    echo -n "TIME: " > "$RESULT"
    date "+%Y-%m-%d %I:%M:%S %p" >> "$RESULT"
    echo "SOURCE: $SOURCE" >> "$RESULT"
    echo >> "$RESULT"
    mkdir -p ./inputs
    mkdir -p "./outputs/tput-$NAME"
    INPUT_MAX_DIM="$INPUT_MAX_DIM" PROG_INPUT=./build/bin/inputgen DIR_INPUT=./inputs DIR_OUTPUT="./outputs/tput-$NAME" "$(dirname "$0")/demo-tput-cuda.sh" "./build/bin/$NAME" "$N_REPS" >> "$DIR_RESULTS/tput-$NAME-$INPUT_MAX_DIM.txt"
  fi
done

shopt -u nullglob
