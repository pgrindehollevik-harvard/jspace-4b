# ORCD readiness and job procedure

This document prepares the **infrastructure** for the Qwen3 scaling extension. It does
not start a new scientific run. The extension needs a version-2 pre-registration before
we select model-specific bands or submit a calibration ladder.

## What is confirmed and what is not

The OnDemand screenshot confirms that the Engaging account exists and is logged in as
`pflo`. That is necessary, but it does not yet prove that Slurm can allocate a GPU to
the account or reveal the queue wait. The ten-minute probe below supplies that evidence.

ORCD's current public documentation says that `mit_normal_gpu` offers L40S and H200 GPUs
with a six-hour maximum job time, while `mit_preemptable` has a 48-hour maximum and may
preempt jobs. ORCD recommends `--requeue` plus checkpointing on the latter. An L40S has
44 GB VRAM and is the right first choice for the 4B and likely 8B compatibility work;
the 32B model requires an H200-class allocation. See ORCD's [scheduler overview](https://orcd-docs.mit.edu/running-jobs/overview/), [resource-request guide](https://orcd-docs.mit.edu/running-jobs/requesting-resources/), and [current public hardware table](https://orcd-docs.mit.edu/running-jobs/available-resources/).

## 1. Inspect the live queue

Open an Engaging shell from the OnDemand portal and run these read-only commands:

```bash
sinfo -O "Partition,Nodes:10,CPUsState,Gres:30,GresUsed:30,StateCompact" -e \
  -p mit_normal_gpu,mit_preemptable
sacctmgr show qos mit_normal_gpu,mit_preemptable format=Name%30,MaxTRESPU%60
squeue --me
```

Copy the output back here. It tells us which GPU types are presently idle or mixed and
what the account is allowed to request. No password, Duo code, SSH private key, or API
key is needed or should be pasted anywhere.

## 2. Bootstrap code and stage the 4B compatibility artifacts

Run these on the **login node**, not in a GPU job. They create a scratch-local checkout,
pin the public Jacobian Lens companion library, and cache only public Hugging Face files.

```bash
export JSPACE_REF=verify-orcd-gpu-jobs
bash <(curl -fsSL https://raw.githubusercontent.com/pgrindehollevik-harvard/jspace-4b/$JSPACE_REF/scripts/orcd/bootstrap.sh)
cd "$HOME/orcd/scratch/jspace-4b"
scripts/orcd/stage_assets.sh qwen3-4b
```

If the shell forbids process substitution, clone the branch first and run the checked-in
script directly:

```bash
git clone --branch verify-orcd-gpu-jobs https://github.com/pgrindehollevik-harvard/jspace-4b.git "$HOME/orcd/scratch/jspace-4b"
cd "$HOME/orcd/scratch/jspace-4b"
scripts/orcd/bootstrap.sh
scripts/orcd/stage_assets.sh qwen3-4b
```

The model and lens are public. The staging command performs no inference and stores no
credentials. It is deliberately done on the login node so compute jobs can set
`HF_HUB_OFFLINE=1` and do not rely on network access.

## 3. Prove GPU allocation, then CUDA compatibility

After bootstrap has created `logs/orcd/`, submit the short probe:

```bash
sbatch scripts/orcd/probe_gpu.sbatch
```

Check it with:

```bash
squeue --me
tail -f logs/orcd/jspace-gpu-probe-<jobid>.out
```

If it completes, record its actual allocation and resource use:

```bash
sacct -j <jobid> -o JobID,JobName,Partition,AllocTRES,Elapsed,State,ExitCode
jobstats <jobid>
```

Then submit the isolated model-and-lens smoke test:

```bash
sbatch scripts/orcd/smoke_cuda.sbatch
```

The smoke test runs one very short clean generation and one ablated generation. It never
touches `results/calibration*.json`, so a hardware check cannot accidentally create
scientific evidence or alter the published 4B record.

## Resource choices after the smoke test

Use one L40S for 4B and 8B work unless measured memory says otherwise. Request an H200
only for models that cannot fit in 44 GB, especially 32B. For an H200 trial, override the
script's default request at submission time:

```bash
sbatch -G h200:1 scripts/orcd/probe_gpu.sbatch
```

Start on `mit_normal_gpu`, which is non-preemptable and easier to debug. Once the
extension's checkpoint and requeue behavior is tested, `mit_preemptable` is appropriate
for long, resumable jobs and must include `#SBATCH --requeue`.

## Portability audit fixed in this branch

The archived Apple-Silicon code had two deployment problems: it defaulted to MPS and
called MPS cache APIs directly, and the public rename changed the documented refit-lens
variable without changing the loader. This branch now automatically chooses CUDA when
available, accepts `JSPACE_DEVICE=cuda`, uses backend-appropriate cache/memory calls,
and standardizes on `JSPACE_LENS_PATH` while retaining `HW0_LENS_PATH` solely for the
historical archived supervisor. The existing result JSONs and paper are intentionally
unchanged.
