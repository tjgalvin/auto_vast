#!/bin/bash -l
#SBATCH --job-name=flint
#SBATCH --export=NONE
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=1-23:00:00
#SBATCH -A OD-207757
#SBATCH --array=1-1%1

#set -e
set -x

# I trust nothing
export OMP_NUM_THREADS=1

export APIURL=http://146.118.68.63:4200/api
export APIURL=http://jones.it.csiro.au:4200/api
export PREFECT_API_URL="${APIURL}"
export WORKDIR=$(pwd)
export PREFECT_HOME="${WORKDIR}/prefect"
export PREFECT_LOGGING_EXTRA_LOGGERS="flint,fixms"
export PREFECT_LOGGING_LEVEL="INFO"
export PREFECT_RESULTS_PERSIST_BY_DEFAULT=true

#echo "Sourcing home"
#source /home/$(whoami)/.bashrc

ENV=$1
echo "Activating conda flint environment: ${ENV}"
conda activate "${ENV}"

DATA="sbids.txt"
cat "$DATA"
echo $SLURM_ARRAY_TASK_ID
LINE=$(cat "${DATA}" | sed -n -e "${SLURM_ARRAY_TASK_ID}p")

SBIDNUM="$LINE"
SBIDDIR="raw/${SBIDNUM}"

# Check if we can skip
if [[ -z $SBIDNUM ]] 
then
    echo "SBID is empty"
    exit 0
fi

if [ -d "archive_copies/$SBIDNUM" ]
then
	echo "$SBIDNUM already exists"
#	exit 0
else
	echo "archive_copies/$SBIDNUM does not exist so will be runing"
fi

if [[ ! -e "$SBIDDIR" ]]
then

    mkdir -p "$SBIDDIR"

    echo "Downloading to $SBIDDIR"

    # Download the data
    vis_download \
        --username drtimgalvin@gmail.com \
	--max-workers 4 \
	--output-dir "$SBIDDIR" \
	--extract-tar \
	--download-holography \
	"${SBIDNUM}"  > /dev/null 2>&1 

else
    echo "Apparently $SBIDDIR exists."
    ls "$SBIDDIR"
fi

# We need to search with find as the path tarballed by ops may be packaged as either
# a fully resolved setonix path, or relatively to ops pipeline execution working directory.
echo "Searching ${SBIDDIR} recursively for linmos cube fits image"
HOLO=$(find "${SBIDDIR}" -type f -name '*cube*fits')

export PREFECT_HOME="${WORKDIR}/prefect_${SBIDNUM}"


if [[ ! -e $HOLO ]]
then
	echo "$HOLO not found. Exiting"
	exit 1
fi

SLEEP=$((RANDOM % 1))
echo "Will wait for ${SLEEP}"
sleep "${SLEEP}"

#flint_flow_continuum_pipeline \
#	"${SBIDDIR}" \
#	--holofile "${HOLO}" \
#	--split-path $(pwd) \
#	--cli-config "cli_config.config" 


WSCLEAN=/scratch3/projects/spiceracs/containers/wsclean_force_mask.sif
WSCLEAN=/scratch3/gal16b/wsclean_scales.sif
YANDA=/scratch3/gal16b/containers/yanda/yandasoft_development_20240819.sif
flint_flow_subtract_cube_pipeline \
	"$(pwd)/${SBIDNUM}" \
        "${WSCLEAN}" \
        "${YANDA}" \
        --holofile "${HOLO}" \
        --cli-config "cli_config_timestep.config"


#mv -v "$(pwd)/${SBIDNUM}/"*contsub*cube*fits "archive_copies/${SBIDNUM}/"
mv -v "$(pwd)/${SBIDNUM}/"*ms "archive_copies/${SBIDNUM}/"

#rm -r "$(pwd)/${SBIDNUM}"
rm casa* *last
rm -r "${PREFECT_HOME}"
#rm -r "${SBIDDIR}"
