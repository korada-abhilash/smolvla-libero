#!/usr/bin/env bash
# Create the conda env used for every run. Usage: CONDA_ROOT=... scripts/setup_env.sh
set -euo pipefail
ENV_NAME=smolvla-libero
source "$CONDA_ROOT/etc/profile.d/conda.sh"

conda create -y -n "$ENV_NAME" python=3.12
conda activate "$ENV_NAME"
conda install -y -c conda-forge ffmpeg=7

# LeRobot 0.6.1 expects CUDA 12.8 torch wheels (works with NVIDIA driver >= 570).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "lerobot[smolvla,libero,training]==0.6.1"
# robosuite logs to a hard-coded /tmp/robosuite.log, which fails on shared nodes where another user
# owns that file. Its documented override, macros_private.py, turns file logging off.
python - <<'PY'
import importlib.util, os, re
d = os.path.dirname(importlib.util.find_spec("robosuite").origin)
src = open(os.path.join(d, "macros.py")).read()
open(os.path.join(d, "macros_private.py"), "w").write(re.sub(r'^FILE_LOGGING_LEVEL = .*$', 'FILE_LOGGING_LEVEL = None', src, flags=re.M))
PY
pip freeze > requirements.lock.txt
