#!/usr/bin/env bash
#SBATCH --job-name=smolvla-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G
#SBATCH --time=23:00:00
#SBATCH --output=logs/%x-%j.log
# Fine-tune SmolVLA (lerobot/smolvla_base) on lerobot/libero, lr 1e-4. VARIANT picks what is trained:
#   full   : the published lerobot/smolvla_libero recipe - vision encoder and VLM trained too, LR decay over 25k
#   expert : smolvla_base's own defaults - vision encoder frozen, only the ~100M action expert trained,
#            LR decay over 30k steps
# With more than one GPU allocated (pass --gres=gpu:N to submit.sh) it runs data-parallel through accelerate;
# BATCH_SIZE is per GPU.
# Usage: VARIANT=full|expert RUN_NAME=... [STEPS=25000 BATCH_SIZE=32 SAVE_FREQ=2500] scripts/submit.sh jobs/train_smolvla.sbatch
#        RESUME=1 RUN_NAME=<same> scripts/submit.sh jobs/train_smolvla.sbatch   # continue from checkpoints/last
set -euo pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate smolvla-libero
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NGPU=$(nvidia-smi -L | wc -l)
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
OUT=outputs/train/$RUN_NAME
case "${VARIANT:?set VARIANT=full or VARIANT=expert}" in
  full)   RECIPE=(--policy.freeze_vision_encoder=false --policy.train_expert_only=false --policy.scheduler_decay_steps=25000) ;;
  expert) RECIPE=(--policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.scheduler_decay_steps=30000) ;;
  *) echo "unknown VARIANT=$VARIANT"; exit 1 ;;
esac
echo "VARIANT=$VARIANT RUN_NAME=$RUN_NAME NGPU=$NGPU STEPS=${STEPS:-25000} BATCH_SIZE(per GPU)=${BATCH_SIZE:-32} RESUME=${RESUME:-0}"

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
    "${RECIPE[@]}" \
    --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
    --num_workers=12 --seed=${SEED:-1000} \
    --output_dir="$OUT"
fi
