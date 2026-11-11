#!/usr/bin/env bash
#SBATCH --job-name=smolvla-eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.log
# LIBERO sim eval of a SmolVLA checkpoint (Hub id or local pretrained_model dir).
# Usage: CKPT=lerobot/smolvla_libero TASK=libero_spatial [N_EPISODES=10] [SEED=1000] RUN_NAME=... \
#          scripts/submit.sh jobs/eval_libero.sbatch
# N_EPISODES is per task (each suite has 10 tasks). TASK may be a comma-separated list of suites.
set -euo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate smolvla-libero
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
# LIBERO asks interactively for its data paths on first import, which a batch job can't answer.
# Write its default config up front (found without importing the package, which is what prompts).
export LIBERO_CONFIG_PATH="$CONDA_PREFIX/.libero"
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  mkdir -p "$LIBERO_CONFIG_PATH"
  python - <<'EOF'
import importlib.util, os, yaml
root = os.path.join(os.path.dirname(importlib.util.find_spec("libero").origin), "libero")
cfg = {"benchmark_root": root, "bddl_files": f"{root}/bddl_files", "init_states": f"{root}/init_files",
       "datasets": os.path.normpath(f"{root}/../datasets"), "assets": f"{root}/assets"}
with open(os.path.join(os.environ["LIBERO_CONFIG_PATH"], "config.yaml"), "w") as f:
    yaml.dump(cfg, f)
print("wrote LIBERO config:", cfg)
EOF
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "CKPT=$CKPT TASK=$TASK N_EPISODES=${N_EPISODES:-10} SEED=${SEED:-1000}"

# LIBERO's cameras are image/image2; SmolVLA checkpoints name them camera1/camera2.
lerobot-eval \
  --policy.path="$CKPT" \
  --env.type=libero --env.task="$TASK" \
  --eval.n_episodes=${N_EPISODES:-10} --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
  --policy.device=cuda --seed=${SEED:-1000} \
  --output_dir=outputs/eval/${RUN_NAME:-eval}_${SLURM_JOB_ID}
