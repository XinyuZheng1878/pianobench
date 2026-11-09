#!/usr/bin/env bash
# Build an expanded candidate set from existing perf/fing files, export and audit.
set -euo pipefail
cd "$(dirname "$0")/.."
WORK=${WORK:-/inspire/hdd/global_user/xuguangjun-253208580299/users/zhengxinyu/shcm_work/20261024}
PY=${PY:-python3}
RUN=${RUN:-$WORK/bench_quality_$(date +%Y%m%d_%H%M%S)}
PER_SUBTYPE=${PER_SUBTYPE:-200}
PASSES=${PASSES:-3}
STRIDE=${STRIDE:-0.5}
export WORK PY PER_SUBTYPE PASSES STRIDE
BENCH="$RUN" bash shcm_bench.sh build
"$PY" scripts/export_bench_gold.py --bench "$RUN" --out "${RUN}_release" "$@"
"$PY" scripts/audit_bench.py --questions "${RUN}_release/questions.jsonl" --out "${RUN}_release/quality_report.json"
printf '\nCandidate report: %s/report.md\nRelease: %s_release\n' "$RUN" "$RUN"
