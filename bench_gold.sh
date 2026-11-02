#!/usr/bin/env bash
# Benchmark on the 11 gold-fingering recordings, run stage by stage (each stage can be rerun).
#
#   bash bench_gold.sh derive      # 第 2 步：导出、八度质检、指法（PMFinger v2 冻结配置）
#
# 环境变量：
#   GOLD   第 1 步 prep_gold.py 的输出目录（默认 ../gold_data）
#   WORK   本流程的工作目录（默认 ../gold_work）
#   WORKERS 指法步骤并行进程数（默认 8）
set -euo pipefail
cd "$(dirname "$0")"
GOLD=${GOLD:-../gold_data}
WORK=${WORK:-../gold_work}
WORKERS=${WORKERS:-8}
PY=${PY:-python3}
FREEZE="kernel=tri sigma_white=0.56 sigma_black=0.56 w_seq=0.5 conf_thresh=0.7 release_lateral=0.35 av_search=12"
step() { echo; echo "▶ $(date '+%H:%M:%S') $*"; }
die() { echo "❌ $*"; exit 1; }

[ -f "${GOLD}/gold_ids.txt" ] || die "找不到 ${GOLD}/gold_ids.txt，先运行第 1 步 scripts/prep_gold.py"
RECS=$(cat "${GOLD}/gold_ids.txt")
N=$(echo ${RECS} | wc -w | tr -d ' ')
[ "${N}" = 11 ] || die "gold_ids.txt 里应有 11 段录音，实际 ${N} 段"
mkdir -p "${WORK}"
PERF_GLOB="${WORK}/perf/[0-9]*[0-9].npz"

derive() {
  step "2a 导出 11 段录音 → ${WORK}/perf"
  ${PY} -B scripts/export_pianovam.py --root "${GOLD}" --out "${WORK}/perf" --recs ${RECS}
  step "2b 八度整段质检"
  ${PY} -B scripts/check_octave.py --inputs "${PERF_GLOB}" --out "${WORK}/qc_octave.json" | tail -n 13
  step "2c 指法与松键（冻结配置） → ${WORK}/fing"
  ${PY} -B scripts/run_fingering.py --inputs "${PERF_GLOB}" --prior models/prior.npz --calib models/calib.npz \
    --out "${WORK}/fing" --workers "${WORKERS}" --set ${FREEZE}
  step "2d 检查结果能被出题规则读取"
  ${PY} -B - "${WORK}" ${RECS} <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from p0_restore import state
work, recs = Path(sys.argv[1]), sys.argv[2:]
bad = []
for vid in recs:
    try:
        P = state.load_raw(work / "perf" / f"{vid}.npz", work / "fing" / f"{vid}.fing.npz")
        print(f"  {vid}  notes {len(P.pitch):5d}  gold fingers {int((P.finger >= 0).sum()):4d}  hand known {int((P.hand >= 0).sum()):5d}")
    except Exception as e:  # noqa: BLE001
        bad.append(vid)
        print(f"  {vid}  FAILED {type(e).__name__}: {e}")
print("✅ 第 2 步完成" if not bad else f"❌ {len(bad)} 段读取失败: {bad}")
sys.exit(1 if bad else 0)
PY
}

case "${1:-}" in
  derive) derive ;;
  *) echo "用法: bash bench_gold.sh derive"; exit 2 ;;
esac
