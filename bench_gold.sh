#!/usr/bin/env bash
# Benchmark on the 11 gold-fingering recordings, run stage by stage (each stage can be rerun).
#
#   bash bench_gold.sh derive      # 第 2 步：导出、八度质检、指法（PMFinger v2 冻结配置）
#   bash bench_gold.sh build       # 第 3 步：出题并渲染帧（不调用任何模型 API）
#   bash bench_gold.sh export      # 第 4 步：导出测评集（答案均衡规则、人工复核筛选、系统提示）
#   bash bench_gold.sh review      # 第 6 步：生成人工复核包（网页 + 缩小的帧），打成 zip
#
# 环境变量：
#   GOLD   第 1 步 prep_gold.py 的输出目录（默认 ../gold_data）
#   WORK   本流程的工作目录（默认 ../gold_work）
#   WORKERS 指法步骤并行进程数（默认 8）
#   PER_SUBTYPE 第 3 步每个子题的题数（11 段合计，默认 20）
#   BENCH  第 3 步输出目录（默认 ${WORK}/bench_v1，必须是新目录）
#   RELEASE 第 4 步输出目录（默认 ${WORK}/release_<BENCH 的名字>，必须是新目录）
#   APPROVALS 人工复核结果文件（jsonl，可多个，用空格分隔）；INCLUDE_UNREVIEWED=1 时连未复核的指法题也导出（只用于试跑测评）
#   EXTEND 在这个已有题库上追加题（已有题原样保留），例如 EXTEND=../gold_work/bench_v1 BENCH=../gold_work/bench_v2 PER_SUBTYPE=50
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

build() {
  BENCH=${BENCH:-${WORK}/bench_v1}
  step "3 出题并渲染帧 → ${BENCH}（每个子题 ${PER_SUBTYPE:-20} 题）"
  ${PY} -B scripts/build_bench_gold.py --work "${WORK}" --gold-ids "${GOLD}/gold_ids.txt" --out "${BENCH}" \
    --per-subtype "${PER_SUBTYPE:-20}" ${EXTEND:+--extend "${EXTEND}"}
  echo "✅ 第 3 步完成：题目 ${BENCH}/questions.jsonl，统计 ${BENCH}/report.md，浏览页 ${BENCH}/index.html"
}

export_release() {
  BENCH=${BENCH:-${WORK}/bench_v1}
  RELEASE=${RELEASE:-${WORK}/release_$(basename "${BENCH}")}
  step "4 导出测评集 ${BENCH} → ${RELEASE}"
  ${PY} -B scripts/export_bench_gold.py --bench "${BENCH}" --out "${RELEASE}" \
    ${APPROVALS:+--approvals ${APPROVALS}} $([ "${INCLUDE_UNREVIEWED:-0}" = 1 ] && echo --include-unreviewed)
  echo "✅ 第 4 步完成：题目 ${RELEASE}/questions.jsonl，浏览页 ${RELEASE}/index.html，导出报告 ${RELEASE}/export_report.md"
  echo "测评：python3 -B scripts/run_models.py --release ${RELEASE} --base-url <接口地址> --model <模型名> --setting F --out ${RELEASE}/runs/<模型名>_F.jsonl"
  echo "评分：python3 -B scripts/evaluate.py --questions ${RELEASE}/questions.jsonl --runs ${RELEASE}/runs/*.jsonl --out ${RELEASE}/report"
}

review() {
  BENCH=${BENCH:-${WORK}/bench_v1}
  REVIEW=${REVIEW:-${WORK}/review_$(basename "${BENCH}")}
  ZIP=${ZIP:-/inspire/hdd/project/project-public/xuguangjun-253208580299/$(basename "${REVIEW}").zip}
  step "6 生成人工复核包 ${BENCH} → ${REVIEW}"
  ${PY} -B scripts/make_review.py --bench "${BENCH}" --out "${REVIEW}" --sample "${SAMPLE:-3}" --zip "${ZIP}"
  echo "✅ 复核包：${ZIP}（在 JupyterLab 文件栏最顶层下载，解压后双击 index.html）"
}

case "${1:-}" in
  derive) derive ;;
  build) build ;;
  export) export_release ;;
  review) review ;;
  *) echo "用法: bash bench_gold.sh derive | build | export | review"; exit 2 ;;
esac
