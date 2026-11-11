#!/usr/bin/env bash
#SBATCH --job-name=smolvla-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x-%j.log
# Fine-tune SmolVLA (lerobot/smolvla_base) on lerobot/libero, following the recipe of the published
# lerobot/smolvla_libero checkpoint (full fine-tune, lr 1e-4, 25k steps). With more than one GPU
# allocated (pass --gres=gpu:N to submit.sh) it runs data-parallel through accelerate; BATCH_SIZE is per GPU.
# Usage: RUN_NAME=... [STEPS=25000 BATCH_SIZE=32 SAVE_FREQ=2500] scripts/submit.sh jobs/train_smolvla.sbatch
#        RESUME=1 RUN_NAME=<same> scripts/submit.sh jobs/train_smolvla.sbatch   # continue from checkpoints/last
set -euo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate smolvla-libero
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
NGPU=$(nvidia-smi -L | wc -l)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
OUT=outputs/train/$RUN_NAME
echo "RUN_NAME=$RUN_NAME NGPU=$NGPU STEPS=${STEPS:-25000} BATCH_SIZE(per GPU)=${BATCH_SIZE:-32} RESUME=${RESUME:-0}"

LAUNCH=lerobot-train
[ "$NGPU" -gt 1 ] && LAUNCH="accelerate launch --multi_gpu --num_processes=$NGPU --mixed_precision=no $(command -v lerobot-train)"

if [ "${RESUME:-0}" = 1 ]; then
  $LAUNCH --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json" --resume=true
else
  $LAUNCH \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=lerobot/libero \
    --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}' \
    --steps=${STEPS:-25000} --batch_size=${BATCH_SIZE:-32} \
    --save_freq=${SAVE_FREQ:-2500} --log_freq=${LOG_FREQ:-100} \
    --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
    --num_workers=12 --seed=${SEED:-1000} \
    --output_dir="$OUT"
fi
