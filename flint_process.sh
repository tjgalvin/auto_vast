#!/bin/bash -l
#SBATCH --job-name=flint_avast_process
#SBATCH --export=NONE
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1-23:00:00
#SBATCH -A OD-241659
#SBATCH --array=1-1%1
#SBATCH --output=flint_avast_process-%A_%a.out
# Usage: sbatch flint_process.sh <conda env> [squash|conda] [submission epoch]

# No partition specified: use the cluster's default compute partition.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from your shared working directory}"
source ./flint_flow_env.sh "$@"

if [[ ! -f "$SBIDDIR/.vis_download_complete" ]]; then
    echo "No successful download marker in $SBIDDIR" >&2
    exit 1
fi

# Retain the original search pattern, but require exactly one match.
HOLO=$(find "$SBIDDIR" -type f -name '*cube*fits')
if [[ -z "$HOLO" || "$HOLO" == *$'\n'* || ! -f "$HOLO" ]]; then
    echo "Expected exactly one holography cube in $SBIDDIR; found:" >&2
    printf '%s\n' "$HOLO" >&2
    exit 1
fi

flint_flow_continuum_pipeline \
    "$SBIDDIR" \
    --holofile "$HOLO" \
    --split-path "$WORKDIR" \
    --cli-config "cli_config.config"

mkdir -p "archive_copies/$SBIDNUM"
shopt -s nullglob
MS_FILES=("$WORKDIR/$SBIDNUM/"*ms)
if (( ${#MS_FILES[@]} == 0 )); then
    echo "No output *ms files found in $WORKDIR/$SBIDNUM" >&2
    exit 1
fi
mv -v -- "${MS_FILES[@]}" "archive_copies/$SBIDNUM/"

# Keep shared-directory casa* and *last files: another task may be using them.
if [[ -d "$PREFECT_HOME" ]]; then
    rm -r -- "$PREFECT_HOME"
fi
