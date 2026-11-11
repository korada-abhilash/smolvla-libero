# SmolVLA Fine-Tuning on LIBERO

**Status: work in progress. No results yet.**

Fine-tunes [SmolVLA](https://huggingface.co/papers/2506.01844) (a 450M-parameter vision-language-action
model) from `lerobot/smolvla_base` on the [LIBERO](https://libero-project.github.io/) simulated
manipulation benchmark with LeRobot, and measures per-suite success (Spatial, Object, Goal, Long) with
confidence intervals. The published `lerobot/smolvla_libero` checkpoint is evaluated under the same
setup as a reproduction reference.

## Setup

Tested with LeRobot 0.6.1, Python 3.12 and CUDA 12.8 PyTorch wheels (NVIDIA driver >= 570).

```bash
CONDA_ROOT=/path/to/miniconda3 bash scripts/setup_env.sh
```

Jobs run under SLURM: copy `cluster.local.env.example` to `cluster.local.env` and fill it in.

```bash
CKPT=lerobot/smolvla_libero TASK=libero_spatial RUN_NAME=published scripts/submit.sh jobs/eval_libero.sbatch
RUN_NAME=smolvla_libero_ft scripts/submit.sh jobs/train_smolvla.sbatch
```

## Results

To be filled in from actual runs.
