#!/bin/bash

if [ "${PROJECT_NAME}" != "difflut" ]; then
    echo "project_variables.sh is not sourced"
    exit 1
fi

# Run the helper_scripts/remote_sync_container.sh to ensure the container is up to date in scratch space
bash helper_scripts/remote_sync_container.sh


# Execute the apptainer container with the appropriate bindings, environment variables, and the given command
apptainer exec \
    --nv \
    --bind "${SCRATCH_STORAGE_DIR}" \
    "${APPTAINER_CONTAINER_SCRATCH}" \
    "$@"
    