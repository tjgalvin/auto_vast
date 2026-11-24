# Sourced by the job_script_prologue of the petrichor*.yaml dask workers.
# Sets FLINT_PYTHON, which the cluster configs use as the worker python, from
# the FLINT_ENV_MODE exported by flint_flow_env.sh
# A cluster config may set FLINT_APPTAINER_MODULE first to pin apptainer

module load "${FLINT_APPTAINER_MODULE:-apptainer}"
unset APPTAINER_BINDPATH
unset SINGULARITY_BINDPATH
export MPLCONFIGDIR=${PWD}

case "${FLINT_ENV_MODE:-conda}" in
squash)
	FLINT_ENV_DIR="${LOCALDIR}/flint_env_${SLURM_JOB_ID}"
	mkdir -p "${FLINT_ENV_DIR}"
	unsquashfs -d "${FLINT_ENV_DIR}" -f flint_squash.sif
	export FLINT_PYTHON="${FLINT_ENV_DIR}/bin/python"
	;;
conda)
	export FLINT_PYTHON="${FLINT_CONDA_PREFIX:-${CONDA_PREFIX}}/bin/python"
	;;
*)
	echo "Unknown FLINT_ENV_MODE=${FLINT_ENV_MODE}"
	exit 1
	;;
esac
