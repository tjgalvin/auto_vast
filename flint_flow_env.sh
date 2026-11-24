# Sourced by the flint_*.sh sbatch scripts, in the working directory that holds
# sbids.txt and the configs. Takes this array task's SBID from sbids.txt,
# activates the environment, and reports the job's timing when it exits
# Usage: source flint_flow_env.sh <conda env> [squash|conda] [submission epoch]
#   squash|conda      how the dask workers get their python, see flint_worker_env.sh
#                     (default conda)
#   submission epoch  Unix time the workflow was submitted, which submit_flint.sh
#                     passes, so the timing includes the queues and the download
# Sets SBIDNUM, SBIDDIR (raw/<SBID>) and WORKDIR for the script that sourced it

usage() {
	echo "Usage: $0 <conda env> [squash|conda] [submission epoch]" >&2
	exit 1
}

[[ $# -ge 1 && $# -le 3 ]] || usage

JOB_STARTED_AT=$(date +%s)
ENV=$1
# Exported so the dask worker jobs, submitted from this job, inherit it
export FLINT_ENV_MODE=${2:-conda}
WORKFLOW_STARTED_AT=${3:-}

case "${FLINT_ENV_MODE}" in
squash)
	[[ -e flint_squash.sif ]] || { echo "flint_squash.sif not found: run flint_env.sh pack <branch>, then flint_env.sh relink <branch> here, or use conda mode" >&2; exit 1; }
	;;
conda) ;;
*)
	usage
	;;
esac

if [[ -n "${WORKFLOW_STARTED_AT}" && ! "${WORKFLOW_STARTED_AT}" =~ ^[1-9][0-9]*$ ]]; then
	echo "The submission epoch must be a positive integer Unix timestamp" >&2
	exit 1
fi

report_timing() {
	local status=$?
	local finished elapsed
	finished=$(date +%s)
	echo "Timing for SBID ${SBIDNUM:-unknown}, task ${SLURM_ARRAY_TASK_ID:-unknown}; exit status ${status}"
	if [[ -n "${WORKFLOW_STARTED_AT}" ]]; then
		elapsed=$((finished - WORKFLOW_STARTED_AT))
		printf 'Submission to completion (including queue and download): %dh %02dm %02ds\n' \
			"$((elapsed / 3600))" "$((elapsed % 3600 / 60))" "$((elapsed % 60))"
	fi
	elapsed=$((finished - JOB_STARTED_AT))
	printf 'Job elapsed: %dh %02dm %02ds\n' \
		"$((elapsed / 3600))" "$((elapsed % 3600 / 60))" "$((elapsed % 60))"
	return "${status}"
}
trap report_timing EXIT

: "${SLURM_ARRAY_TASK_ID:?Submit as an array job, one task per line of sbids.txt}"
SBIDNUM=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sbids.txt)
SBIDNUM=${SBIDNUM%$'\r'}
if [[ ! "${SBIDNUM}" =~ ^[0-9]+$ ]]; then
	echo "Missing or invalid SBID at line ${SLURM_ARRAY_TASK_ID} of sbids.txt" >&2
	exit 1
fi
SBIDDIR="raw/${SBIDNUM}"

echo "Activating conda flint environment: ${ENV}, workers use ${FLINT_ENV_MODE}"
# conda must be initialised by your login setup, which #!/bin/bash -l runs
conda activate "${ENV}" || exit 1
export FLINT_CONDA_PREFIX="${CONDA_PREFIX}"

# I trust nothing
export OMP_NUM_THREADS=1

export WORKDIR="${PWD}"
export PREFECT_API_URL=http://tones.it.csiro.au:4200/api
# One prefect directory per SBID, so array tasks never share one
export FLINT_PREFECT_DIR="${WORKDIR}/prefect_${SBIDNUM}"
export PREFECT_HOME="${FLINT_PREFECT_DIR}"
export PREFECT_RESULTS_LOCAL_STORAGE_PATH="${FLINT_PREFECT_DIR}"
export PREFECT_LOGGING_EXTRA_LOGGERS="flint,fixms,fitscube,eye-patch"
export PREFECT_LOGGING_LEVEL="INFO"
export PREFECT_RESULTS_PERSIST_BY_DEFAULT=true
export PREFECT_API_REQUEST_TIMEOUT=120
export PREFECT_CLIENT_MAX_RETRIES=8
export PREFECT_LOGGING_TO_API_BATCH_INTERVAL=10
