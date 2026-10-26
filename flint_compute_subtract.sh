#!/bin/bash -l
#SBATCH --job-name=flint_compute_subtract
#SBATCH --export=NONE
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1-23:00:00
#SBATCH -A OD-241659
#SBATCH --array=1-1%1
#SBATCH --output=flint_compute_subtract-%A_%a.out

# No partition specified: use the cluster's default compute partition.
set -eo pipefail
ENV=${1:?Usage: sbatch flint_compute_subtract.sh CONDA_ENV}
cd "${SLURM_SUBMIT_DIR:?Submit from your shared working directory}"
export OMP_NUM_THREADS=1
export APIURL=http://tones.it.csiro.au:4200/api
export PREFECT_API_URL="$APIURL"
export WORKDIR="$PWD"
export PREFECT_LOGGING_EXTRA_LOGGERS="flint,fixms"
export PREFECT_LOGGING_LEVEL="INFO"
export PREFECT_RESULTS_PERSIST_BY_DEFAULT=true
export PREFECT_API_REQUEST_TIMEOUT=120
export PREFECT_CLIENT_MAX_RETRIES=8
export PREFECT_LOGGING_TO_API_BATCH_INTERVAL=10
# As in the original script, conda must be initialized by your login setup.
conda activate "$ENV"

SBIDNUM=$(sed -n "${SLURM_ARRAY_TASK_ID:?}p" sbids.txt)
SBIDNUM=${SBIDNUM%$'\r'}
if [[ ! "$SBIDNUM" =~ ^[0-9]+$ ]]; then
    echo "Missing or invalid SBID at line $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi
SBIDDIR="raw/$SBIDNUM"
export PREFECT_HOME="$WORKDIR/prefect_${SBIDNUM}"
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

# Container paths 
# Alternative WSCLEAN: /scratch3/projects/spiceracs/containers/wsclean_force_mask.sif
WSCLEAN=/scratch3/ros314/containers/wsclean_scales.sif
YANDA=/scratch3/ros314/containers/flint-containers_askapsoft.sif
for required in "$WSCLEAN" "$YANDA" cli_config_timestep.config; do
    if [[ ! -r "$required" ]]; then
        echo "Required subtraction input is not readable: $required" >&2
        exit 1
    fi
done

flint_flow_continuum_pipeline \
    "$SBIDDIR" \
    --holofile "$HOLO" \
    --split-path "$WORKDIR" \
    --cli-config "cli_config.config"

flint_flow_subtract_cube_pipeline \
    "$WORKDIR/$SBIDNUM" \
    "$WSCLEAN" \
    "$YANDA" \
    --holofile "$HOLO" \
    --cli-config "cli_config_timestep.config"

mkdir -p "archive_copies/$SBIDNUM"
shopt -s nullglob
CUBE_FILES=("$WORKDIR/$SBIDNUM/"*contsub*cube*fits)
if (( ${#CUBE_FILES[@]} == 0 )); then
    echo "No output *contsub*cube*fits files found in $WORKDIR/$SBIDNUM" >&2
    exit 1
fi
mv -v -- "${CUBE_FILES[@]}" "archive_copies/$SBIDNUM/"

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
