#!/bin/bash
# Run with bash from the shared working directory containing sbids.txt, the
# configs and the flint_*_env.sh scripts. Do not submit this wrapper with sbatch.
# Usage: ./submit_flint.sh [array] [conda env] [squash|conda]
#   array      sbids.txt lines to process, as for sbatch --array (default 1-1%1)
#   conda env  default flint_main
#   squash|conda  how the dask workers get their python (default conda)
set -eo pipefail

ARRAY="${1:-1-1%1}"
ENV="${2:-flint_main}"
MODE="${3:-conda}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# The sbatch scripts and dask workers source the env scripts from the working directory
for required in sbids.txt cli_config.config flint_flow_env.sh flint_worker_env.sh; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required in working directory: $PWD" >&2
        exit 1
    fi
done
if [[ "$MODE" == squash && ! -e flint_squash.sif ]]; then
    echo "flint_squash.sif not found: run flint_env.sh pack <branch>, then flint_env.sh relink <branch> here, or use conda mode" >&2
    exit 1
fi

WORKFLOW_STARTED_AT=$(date +%s)
echo "Workflow submitted at: $(date)"

download_id=$(sbatch --parsable \
    --array="$ARRAY" \
    "$SCRIPT_DIR/flint_download.sh" "$ENV" "$MODE")
download_id=${download_id%%;*}
echo "Submitted download array: $download_id"

if compute_id=$(sbatch --parsable \
    --array="$ARRAY" \
    --dependency="aftercorr:$download_id" \
    "$SCRIPT_DIR/flint_process_subtract.sh" "$ENV" "$MODE" "$WORKFLOW_STARTED_AT"); then

    #"$SCRIPT_DIR/flint_process.sh" "$ENV" "$MODE" "$WORKFLOW_STARTED_AT"); then
    echo "Submitted compute array: ${compute_id%%;*} (waiting for matching downloads)"
    echo "Timing will be reported in flint_avast_process_subtract-${compute_id%%;*}_<task>.out"

else
    echo "Compute submission failed; download array $download_id remains submitted." >&2
    exit 1
fi
