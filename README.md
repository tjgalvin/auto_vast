Scripts to process the continuum imaging for VAST data to perform subsequent time step imaging
(Updated for EMU testing)

The layout follows
[racs-all-configuration](https://github.com/tjgalvin/racs-all-configuration):
the sbatch scripts are thin and source `flint_flow_env.sh`, and the dask
workers in `petrichor*.yaml` source `flint_worker_env.sh`.

## Files

| File | What it does | Used by |
|------|--------------|---------|
| `submit_flint.sh` | Starts a run: submits the download array, then the compute array | you |
| `flint_download.sh` | Downloads each SBID from CASDA into `raw/<SBID>/` | `submit_flint.sh` |
| `flint_process_subtract.sh` | Continuum imaging, then continuum subtraction and time step imaging | `submit_flint.sh` |
| `flint_process.sh` | Continuum imaging only | you, by hand |
| `flint_flow_env.sh` | Takes the task's SBID from `sbids.txt`, activates conda, sets up Prefect, reports timing | each sbatch script |
| `flint_worker_env.sh` | Loads apptainer and picks the dask workers' python | the dask workers |
| `cli_config.config` | flint options for the continuum pipeline | `flint_flow_continuum_pipeline` |
| `cli_config_timestep.config` | flint options for the subtraction pipeline | `flint_flow_subtract_cube_pipeline` |
| `petrichor.yaml` | dask workers for the continuum pipeline | `cli_config.config` |
| `petrichor_timestep.yaml` | dask workers for the subtraction pipeline | `cli_config_timestep.config` |
| `emu_deep.yaml` | Imaging strategy | both cli configs |
| `flint_env.sh` | Builds and packs flint conda environments (optional) | you |
| `sbids.txt` | Your SBIDs, one per line. Not tracked by git | the sbatch scripts |

## How a run fits together

```
./submit_flint.sh 1-4%2
 │
 ├─ 1. sbatch flint_download.sh            array on the io partition, one task per sbids.txt line
 │       └─ vis_download → raw/<SBID>/, then raw/<SBID>/.vis_download_complete
 │
 └─ 2. sbatch flint_process_subtract.sh    array, --dependency=aftercorr:<download array>
         │                                 task N starts once download task N has succeeded
         ├─ flint_flow_continuum_pipeline      cli_config.config, workers from petrichor.yaml
         ├─ flint_flow_subtract_cube_pipeline  cli_config_timestep.config, workers from petrichor_timestep.yaml
         └─ move *contsub*cube*fits and *ms → archive_copies/<SBID>/, remove prefect_<SBID>/
```

Every sbatch job sources `flint_flow_env.sh` first, and every dask worker job
sources `flint_worker_env.sh`. Both are read from the working directory, so
run from this checkout, where the configs also live.

## Setting up

1. Work in this checkout on Petrichor. The run products (`raw/`,
   `archive_copies/`, `flint_logs/`, `prefect_*/`) are written here and are
   ignored by git.
2. Have a conda environment with flint and `vis_download` installed.
   `flint_main` is the default name. The configs need a flint with eye-patch v2
   masking: flint main once PR
   [#471](https://github.com/flint-crew/flint/pull/471) is merged, or its branch
   until then (see [Testing a flint branch](#testing-a-flint-branch)). An older
   `flint_main` must be updated first. conda must be initialised by your login
   setup, as the sbatch scripts run `bash -l`.
3. Check the paths and account for your setup:
   - the containers in `cli_config.config`, and `WSCLEAN` and `YANDA` in
     `flint_process_subtract.sh`
   - the CASDA `--username` in `flint_download.sh`
   - the account (`-A` / `account:`) in the sbatch headers and `petrichor*.yaml`
4. List the SBIDs in `sbids.txt`, one per line.

## Running

```bash
./submit_flint.sh [array] [conda env] [squash|conda]
./submit_flint.sh 1-4%2                       # lines 1-4 of sbids.txt, two at a time
./submit_flint.sh 5-5 flint_main conda        # line 5 only
```

`array` selects lines of `sbids.txt`, as for `sbatch --array`. The defaults are
`1-1%1`, `flint_main` and `conda` (see [Worker modes](#worker-modes)). Run it
with bash, not sbatch.

Where to look:

| What | Where |
|------|-------|
| Download jobs | `flint_avast_download-<job>_<task>.out` |
| Compute jobs, with the timing report at the end | `flint_avast_process_subtract-<job>_<task>.out` |
| dask workers | `flint_logs/` |
| Flow runs | the Prefect UI on tones |
| Products | `archive_copies/<SBID>/` (flint's `--sbid-copy-path`, plus the moved cubes and MSs) |

Each job reports its run time when it exits. `submit_flint.sh` records the time
before it submits the downloads and passes it to the compute jobs, so they also
report the time since submission, including the queues and the download.

## Re-running and failures

- A download is skipped if `raw/<SBID>/.vis_download_complete` exists. If
  `raw/<SBID>/` exists without it, from a failed or older download, the task
  fails: inspect the directory and move it aside before trying again.
- If download task N fails, compute task N never starts. It is left pending
  (`DependencyNeverSatisfied`); cancel it with `scancel`.
- Compute jobs are not skipped when `archive_copies/<SBID>/` already exists.
  Resubmitting the same lines runs them again, but flint refuses to start while
  its output folder `<SBID>/` exists: remove it first.
- If the continuum flow fails, the job stops before the subtraction.
- Up to `--max-failed-intervals` (10, in `cli_config_timestep.config`) time steps
  that fail are left out of the cube, and each is logged. More than that fails
  the subtraction flow.
- `prefect_<SBID>/` is only removed when the compute job succeeds.

## Running the steps by hand

The scripts take `<conda env> [squash|conda] [submission epoch]`. The download
has to come first: the compute scripts fail without the download marker.

```bash
# Continuum only, waiting for the downloads, as submit_flint.sh does
dl=$(sbatch --parsable --array=1-4 flint_download.sh flint_main conda)
sbatch --array=1-4 --dependency=aftercorr:${dl%%;*} flint_process.sh flint_main conda

# Once raw/<SBID>/ has its download marker, a compute script can run alone
sbatch --array=3-3 flint_process_subtract.sh flint_main conda
```

To make `submit_flint.sh` run the continuum only, swap the commented
`flint_process.sh` line in for the `flint_process_subtract.sh` one.

## Worker modes

How the dask workers get their python, set by the `[squash|conda]` argument:

- `conda` (default): workers run straight from the conda environment, as before
  this layout. Nothing to set up.
- `squash`: each worker unpacks `flint_squash.sif`, a symlink to
  `flint_<branch>/<branch>-conda.squashfs`, to its local disk. Lighter on the
  shared filesystem, but the squashfs has to be repacked after every change to
  the environment. Make it with `flint_env.sh pack` and `relink`, below.

## Environments

`flint_env.sh <action> <branch>` manages the `flint_<branch>` environments, as
in racs-all-configuration. The `flint_<branch>` repos and their squashfs are
kept in `FLINT_REPOS_DIR`, which defaults to the current directory:

```bash
FLINT_REPOS_DIR=/path/to/flint_repos ./flint_env.sh pack main
./flint_env.sh relink main
```

| action    | does                                                        |
|-----------|-------------------------------------------------------------|
| `create`  | clone flint, check out the branch, build the environment    |
| `install` | pull and reinstall flint, no packing (enough for `conda`)   |
| `update`  | `install`, then pack the squashfs (needed for `squash`)     |
| `develop` | editable install, `conda` mode only                         |
| `pack`    | pack the squashfs                                           |
| `relink`  | point `flint_squash.sif` here at the branch's squashfs      |
| `delete`  | remove the environment and its directory                    |

`install` and `update` pull the latest flint of the branch, so once
[#471](https://github.com/flint-crew/flint/pull/471) is merged,
`./flint_env.sh update main` brings `flint_main` up to date. `pack` and
`relink` leave flint alone.

## Testing a flint branch

Each flint branch gets its own environment, `flint_<branch>`. To try the branch
of PR [#471](https://github.com/flint-crew/flint/pull/471) before it is merged:

```bash
FLINT_REPOS_DIR=/path/to/flint_repos ./flint_env.sh create subtract-flow-robustness
conda activate flint_subtract-flow-robustness
pip install git+https://github.com/AlecThomson/vis-downloader   # the download job runs in this environment too
flint_config verify emu_deep.yaml     # should report that it appears valid
./submit_flint.sh 1-1 flint_subtract-flow-robustness
```

- `./flint_env.sh install subtract-flow-robustness` picks up later pushes to the
  branch.
- Verify the strategy before submitting. flint creates `<SBID>/` before it
  checks the strategy, so a rejected strategy leaves it behind, and the next run
  will not start until it is removed.
- Each flow run writes its dask performance report to
  `flint_logs/dask-performance-report-<flow run>.html`.
