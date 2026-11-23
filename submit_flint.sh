#!/bin/bash
# Run with bash from the shared working directory containing sbids.txt
# and cli_config.config. Do not submit this wrapper with sbatch.
set -eo pipefail

ENV="flint_main"
ARRAY="${1:-1-1%1}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

for required in sbids.txt cli_config.config; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required in working directory: $PWD" >&2
        exit 1
    fi
done

WORKFLOW_STARTED_AT=$(date +%s)
echo "Workflow submitted at: $(date)"

download_id=$(sbatch --parsable \
    --array="$ARRAY" \
    "$SCRIPT_DIR/flint_download.sh" "$ENV")
download_id=${download_id%%;*}
echo "Submitted download array: $download_id"

if compute_id=$(sbatch --parsable \
    --array="$ARRAY" \
    --dependency="aftercorr:$download_id" \
    "$SCRIPT_DIR/flint_process_subtract.sh" "$ENV" "$WORKFLOW_STARTED_AT"); then

    #"$SCRIPT_DIR/flint_process.sh" "$ENV"); then
    echo "Submitted compute array: ${compute_id%%;*} (waiting for matching downloads)"
    echo "Timing will be reported in flint_avast_process_subtract-${compute_id%%;*}_<task>.out"

else
    echo "Compute submission failed; download array $download_id remains submitted." >&2
    exit 1
fi
