#!/bin/bash

# Execute the apptainer container with the appropriate bindings, environment variables, and the given command
apptainer exec \
    --nv \
    "difflut.sif" \
    "$@"
 