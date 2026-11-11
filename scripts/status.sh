#!/usr/bin/env bash
# One-screen status: queued/running jobs, latest training step, finished LIBERO evals. Run on the cluster login node.
cd "$(dirname "$0")/.."
squeue -u "$USER" -o "%.8i %.14j %.9T %.10M %R"
echo; for f in $(ls -t logs/smolvla-train-*.log 2>/dev/null | head -1); do echo "$f: $(grep -o 'step:[^ ]* .*loss:[0-9.]*' "$f" | tail -1 | cut -c1-60)"; done
echo; echo "LIBERO evals (success %, 10 episodes per task):"
for d in outputs/eval/*/; do
  [ -f "$d/eval_info.json" ] || continue
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]+'/eval_info.json'));g=d['per_group'];print(f\"  {sys.argv[2]:34s} \"+'  '.join(f\"{k.replace('libero_','')}={v['pc_success']:.0f}\" for k,v in g.items()))" "$d" "$(basename "$d")"
done
