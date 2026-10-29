#!/usr/bin/env bash
# 钢琴手部视频问答 Benchmark：从原始数据到出分，8 步一条命令跑完（每步可单独重跑）。
#
#   bash run_all.sh                          # 步骤 1-7：导出、质检、指法、状态、出题、复核图、渲染
#   BASE_URL=http://localhost:8001/v1 MODEL=Qwen/Qwen3-VL-8B-Instruct TAG=qwen bash run_all.sh   # 再加步骤 8
#
# 环境变量：
#   DATA      PianoVAM 根目录（含 metadata.json、Video/、Handskeleton/、TSV/、Fingering/、Fingering_GT/）
#   WORK      输出目录（默认 ./work）
#   PER_TASK  每个任务的事实数（试点 30，正式约 600）
#   RECS      只处理这些录音（空格分隔，默认全部）
#   SETTINGS  步骤 8 跑哪些设置（默认 "F blind"；有视频输入能力的模型可加 V）
#   MAX_SIDE  推理时把帧缩到的长边像素（显存紧张时用，例如 896）
set -euo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-data_pianovam}
WORK=${WORK:-work}
PER_TASK=${PER_TASK:-30}
RECS=${RECS:-}
SETTINGS=${SETTINGS:-"F blind"}
MAX_SIDE=${MAX_SIDE:-0}
PY=${PY:-python3}
FREEZE="kernel=tri sigma_white=0.56 sigma_black=0.56 w_seq=0.5 conf_thresh=0.7 release_lateral=0.35 av_search=12"
mkdir -p "${WORK}"
step() { echo; echo "▶ $*"; }

step "1/8 数据导出 → ${WORK}/perf"
if [ -n "${RECS}" ]; then
  ${PY} -B scripts/export_pianovam.py --root "${DATA}" --out "${WORK}/perf" --recs ${RECS}
else
  ${PY} -B scripts/export_pianovam.py --root "${DATA}" --out "${WORK}/perf"
fi
PERF_GLOB="${WORK}/perf/[0-9]*[0-9].npz"

step "2/8 质检：八度整段检查"
${PY} -B scripts/check_octave.py --inputs "${PERF_GLOB}" --out "${WORK}/qc_octave.json"

step "3/8 指法与松键（PMFinger v2 冻结配置）"
${PY} -B scripts/run_fingering.py --inputs "${PERF_GLOB}" --prior models/prior.npz --calib models/calib.npz \
  --out "${WORK}/fing" --workers "${WORKERS:-4}" --set ${FREEZE}

step "4/8 状态派生检查"
${PY} -B scripts/check_states.py --perf "${PERF_GLOB}" --fing "${WORK}/fing" --out "${WORK}/qc_states.json"

step "5/8 出题（每任务 ${PER_TASK} 个事实）"
${PY} -B scripts/gen_bench.py --perf "${PERF_GLOB}" --fing "${WORK}/fing" --out "${WORK}/bench" --per-task "${PER_TASK}"

step "6-7/8 渲染片段、帧序列、标注和复核图"
${PY} -B scripts/render_clips.py --questions "${WORK}/bench/questions.jsonl" --perf "${WORK}/perf" \
  --fing "${WORK}/fing" --out "${WORK}/release" --ablation-fps 2 4 --review
echo "人工复核：打开 ${WORK}/release/review/*.jpg，在 ${WORK}/bench/review.tsv 的 keep 列填 1/0"

step "扩写准备：生成给大模型的事实卡"
${PY} -B scripts/make_cards.py --bench "${WORK}/bench" --release "${WORK}/release" --perf "${WORK}/perf" \
  --fing "${WORK}/fing" --out "${WORK}/cards.jsonl"
echo "扩写：python -B scripts/expand_qa.py run --cards ${WORK}/cards.jsonl --release ${WORK}/release --out ${WORK}/expansions.jsonl --base-url ... --model ..."

if [ -n "${BASE_URL:-}" ]; then
  step "8/8 推理与评分：${MODEL}"
  RUNS=()
  for S in ${SETTINGS}; do
    OUT="${WORK}/runs/${TAG:-model}_${S}.jsonl"
    ${PY} -B scripts/run_models.py --release "${WORK}/release" --base-url "${BASE_URL}" --model "${MODEL}" \
      --tag "${TAG:-model}" --setting "${S}" --out "${OUT}" --max-side "${MAX_SIDE}" --workers "${INFER_WORKERS:-4}"
    RUNS+=("${OUT}")
  done
  ALL=$(ls "${WORK}"/runs/*.jsonl 2>/dev/null || true)
  ${PY} -B scripts/evaluate.py --questions "${WORK}/release/questions.jsonl" --runs ${ALL} --out "${WORK}/report"
  echo "报告：${WORK}/report/report.md"
fi
