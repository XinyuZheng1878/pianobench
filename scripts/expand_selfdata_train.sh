#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
work=${WORK:-/inspire/hdd/global_user/xuguangjun-253208580299/users/zhengxinyu/shcm_work/20261024}
out=${OUT:-$work/train_expanded_$(date +%Y%m%d_%H%M%S)}
python3 -B scripts/build_bench_gold.py \
  --work "$work" --gold-ids "$work/ids.txt" \
  --exclude-test "$work/bench_1105_0638_release_big2/questions.jsonl" \
  --out "$out" --training-expansion --short-windows \
  --stride 0.25 --passes 8 --per-subtype "${PER_SUBTYPE:-1000}"
python3 -B scripts/export_bench_gold.py --bench "$out" --out "${out}_release"
wc -l "${out}_release/questions.jsonl"
