#!/usr/bin/env bash
# HISTORICAL TOOLING: incident repair, NOT current revised-paper reproduction.
# WARNING: launches remote benchmark jobs and can overwrite stored source files,
# including results/boolq_{mistral7b,granite8b}_rerun.json used by the revised paper.
# The historical SAFETY note below does not protect existing *_rerun.json files.
# For current table reproduction use scripts/build_descriptive_tables.py;
# see RESULTS.md and RESULTS_HISTORICAL.md. Do not run this as a paper rebuild.
# rerun_canonical.sh — launch the minimal set of canonical re-runs that make
# every Table 1 cell unassailable.
#
# Three gaps closed (audit: scripts/verify_tables.py + the vote/mean scan):
#   1. RuleTaker: loop was scored on 400/1000 val rows; readout is full 1000.
#      Re-run loop on the FULL 1000-row val so readout vs loop sit on
#      identical rows (last.mlp, last.mlp.loop_matched, loop arms, and the
#      paired _rowpreds all land on the same 1000 rows). All 6 models.
#   2. BoolQ Mistral/Granite: loop was on 405/3270 rows -> full 3270.
#   3. BoolQ Qwen3-4B: the table value (0.862) was not reproducible from any
#      artifact (mean 0.857, vote 0.858, paired 0.8615). Fresh budget-style
#      run, full 3270 val, k 0,8,16,32,64, loop pad 8192.
#
# Protocols identical to the canonical runs EXCEPT the one gap (loop_val), so
# results stay comparable. Hidden vectors + trained heads come from the
# existing caches; only the loop arm (and its rowpreds) is recomputed.
#
# SAFETY: promotes into results/*_rerun.json (NOT the canonical names). The
# canonical files are promoted by hand after scripts/verify_tables.py agrees.
#
# Usage: bash scripts/rerun_canonical.sh [loop_only]
#   loop_only  -> only the 8 loop runs (skip the BoolQ 4B full re-run)
set -uo pipefail
cd "$(dirname "$0")/.."

LOGDIR=logs/rerun_$(date +%Y%m%dT%H%M%S)
mkdir -p "$LOGDIR"
echo "logging to $LOGDIR"

launch() { # task model gpu max_train max_val loop_val loop_pad k_shots promote logname
  local task="$1" model="$2" gpu="$3" mtr="$4" mva="$5" lva="$6" lpad="$7" k="$8" promote="$9" log="${10}"
  echo "==> $log  ($task $model gpu=$gpu loop_val=$lva)"
  nohup modal run --detach bench.py --task "$task" --model "$model" --gpu "$gpu" \
      --max-train "$mtr" --max-val "$mva" --loop-val "$lva" --batch 8 \
      --loop-pad-max "$lpad" --k-shots "$k" --promote "$promote" \
      > "$LOGDIR/$log.log" 2>&1 &
  echo "    pid=$!  log=$LOGDIR/$log.log"
}

MODE="${1:-all}"

# ---- RuleTaker: loop on FULL 1000-val (was 400) ----
# 0.6B done as smoke test (run 20260811T111425_29ce35); skip.
# launch ruletaker "Qwen/Qwen3-0.6B"  h200 2000 1000 1000 2048 "0,8" ruletaker_qwen06b_n2k_rerun.json  rt_qwen06b
launch ruletaker "Qwen/Qwen3-4B"    h200 2000 1000 1000 2048 "0,8" ruletaker_qwen4b_n2k_rerun.json    rt_qwen4b
launch ruletaker "Qwen/Qwen3-8B"    b200 2000 1000 1000 2048 "0,8" ruletaker_qwen8b_n2k_rerun.json    rt_qwen8b
launch ruletaker "mistralai/Mistral-7B-v0.3" b200 2000 1000 1000 2048 "0,8" ruletaker_mistral7b_n2k_rerun.json rt_mistral7b
launch ruletaker "ibm-granite/granite-3.1-8b-base" b200 2000 1000 1000 2048 "0,8" ruletaker_granite8b_n2k_rerun.json rt_granite8b
launch ruletaker "deepseek-ai/DeepSeek-V4-Flash-0731" b300 2000 1000 1000 2048 "0,8" ruletaker_dsv4_n2k_rerun.json rt_dsv4

# ---- BoolQ Mistral/Granite: loop on FULL 3270-val (was 405) ----
launch boolq "mistralai/Mistral-7B-v0.3" b200 9427 3270 3270 2048 "0,8" boolq_mistral7b_rerun.json  bq_mistral7b
launch boolq "ibm-granite/granite-3.1-8b-base" b200 9427 3270 3270 2048 "0,8" boolq_granite8b_rerun.json bq_granite8b

# ---- BoolQ Qwen3-4B: fresh budget-style run (ghost cell) ----
if [ "$MODE" != "loop_only" ]; then
  launch boolq "Qwen/Qwen3-4B" h200 9427 3270 3270 8192 "0,8,16,32,64" boolq_budget_4b_rerun.json bq_4b
fi

echo
echo "launched. Monitor:  modal app list   (Tasks=0 => done)"
echo "recover:  scripts/vol get <slug>/<task>/runs/<run_id>.json -d results/"
echo "logs:     tail -f $LOGDIR/*.log"
