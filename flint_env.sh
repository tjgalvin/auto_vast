#!/usr/bin/env bash
# Usage: flint_env.sh <create|install|update|develop|pack|relink|delete> <branch>
#   install: pull and reinstall, without packing. Enough for conda mode workers
#   update:  install, then pack the squashfs used by squash mode workers
#   develop: editable install, only usable by conda mode workers
#   relink:  point flint_squash.sif in the current directory at the branch's squashfs
# The flint_<branch> repos and their squashfs live in FLINT_REPOS_DIR, which
# defaults to the current directory, e.g.
#   FLINT_REPOS_DIR=/scratch3/gal16b/flint_envs flint_env.sh update main

usage() {
	echo "Usage: $0 <create|install|update|develop|pack|relink|delete> <branch>"
	exit 1
}

[[ $# -eq 2 ]] || usage

ACTION=$1
BRANCH=${2%/}
BRANCH=${BRANCH#flint_}
DIR="flint_${BRANCH}"
# Absolute, as the actions cd into the repo before packing
FLINT_REPOS_DIR=$(cd "${FLINT_REPOS_DIR:-.}" && pwd) || exit
REPO_DIR="${FLINT_REPOS_DIR}/${DIR}"
SQUASHFS="${REPO_DIR}/${BRANCH}-conda.squashfs"

squash() {
	# Remove rather than overwrite, so jobs with the old image mounted keep running
	rm -f "${SQUASHFS}" && \
		conda pack -n "${DIR}" -o "${SQUASHFS}" --format squashfs -j -1
}

set -x
source /home/$(whoami)/.bashrc

case "${ACTION}" in
create)
	mkdir "${REPO_DIR}" || exit
	cd "${REPO_DIR}" || exit

	git clone git@github.com:tjgalvin/flint.git && \
		cd flint && \
		git checkout "${BRANCH}" || exit

	# Not editable: conda pack refuses environments with editable installs
	conda create -y -n "${DIR}" python=3.12 && \
		conda activate "${DIR}" && \
		conda install -y -c conda-forge boost casacore python-casacore compilers uv numba conda-pack squashfs-tools && \
		uv pip install '.[all]'
	;;
install|update)
	cd "${REPO_DIR}/flint" || exit

	# uv only rebuilds a local package when pyproject.toml changes, so force it
	git pull && \
		conda activate "${DIR}" && \
		uv pip install --reinstall-package askap-flint '.[all]' || exit

	[[ "${ACTION}" == update ]] && squash
	;;
develop)
	cd "${REPO_DIR}/flint" || exit

	conda activate "${DIR}" && \
		uv pip install -e .
	;;
pack)
	conda activate "${DIR}" && \
		squash
	;;
relink)
	[[ -f "${SQUASHFS}" ]] || { echo "${SQUASHFS} does not exist, run pack first"; exit 1; }
	ln -sfn "${SQUASHFS}" flint_squash.sif
	;;
delete)
	conda remove -y -n "${DIR}" --all
	rm -rf "${REPO_DIR}"
	;;
*)
	usage
	;;
esac
