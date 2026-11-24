#!/bin/bash -l
#SBATCH --job-name=flint_avast_process_subtract
#SBATCH --export=NONE
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1-23:00:00
#SBATCH -A OD-241659
#SBATCH --array=1-1%1
#SBATCH --output=flint_avast_process_subtract-%A_%a.out
# Usage: sbatch flint_process_subtract.sh <conda env> [squash|conda] [submission epoch]

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

# Container paths 
# Alternative WSCLEAN: /scratch3/projects/spiceracs/containers/wsclean_force_mask.sif
#WSCLEAN=/scratch3/ros314/containers/wsclean_scales.sif
WSCLEAN=/scratch3/ros314/containers/wsclean-znver3-wsclean-speedups-2a3fea68.sif
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
