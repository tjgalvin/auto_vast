#!/bin/bash -l
#SBATCH --job-name=flint_download
#SBATCH --export=NONE
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1-23:00:00
#SBATCH -A OD-241659
#SBATCH --array=1-1%1
#SBATCH --partition=io
#SBATCH --output=flint_download-%A_%a.out

set -eo pipefail
ENV=${1:?Usage: sbatch flint_download.sh CONDA_ENV}
cd "${SLURM_SUBMIT_DIR:?Submit from your shared working directory}"
export OMP_NUM_THREADS=1
# As in the original script, conda must be initialized by your login setup.
conda activate "$ENV"

SBIDNUM=$(sed -n "${SLURM_ARRAY_TASK_ID:?}p" sbids.txt)
SBIDNUM=${SBIDNUM%$'\r'}
if [[ ! "$SBIDNUM" =~ ^[0-9]+$ ]]; then
    echo "Missing or invalid SBID at line $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi
SBIDDIR="raw/$SBIDNUM"
MARKER="$SBIDDIR/.vis_download_complete"

if [[ -f "$MARKER" ]]; then
    echo "Previously completed download: $SBIDDIR"
else
    # Do not silently trust directories left by the old script or a failed run.
    if [[ -e "$SBIDDIR" ]]; then
        echo "$SBIDDIR exists without a completion marker." >&2
        echo "Inspect it and move it aside before retrying this download." >&2
        exit 1
    fi
    mkdir -p "$SBIDDIR"
    vis_download \
        --username kros9503@uni.sydney.edu.au \
        --max-workers 12 \
        --output-dir "$SBIDDIR" \
        --extract-tar \
        --download-holography \
        "$SBIDNUM"
    touch "$MARKER"
fi
