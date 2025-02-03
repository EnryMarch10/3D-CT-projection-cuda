#!/bin/bash

terminate_tree() {
    local parent_pid=$1
    # Get all child PIDs of the given process
    local children=$(pgrep -P $parent_pid)
    for child in $children; do
        terminate_tree $child  # Recursively terminate child processes
    done
    # Kill the parent process
    kill -9 $parent_pid
}

PID=$(pgrep -f ./scripts/tests.sh)

if [ ! -d "/proc/$PID" ]; then
    echo "PID '$PID' is not valid or the process is not running" >&2
fi

# Start termination with the daemon script PID
terminate_tree $PID
