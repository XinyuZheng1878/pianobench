#!/usr/bin/env bash
# 上音自采数据（SHCM）跑原有出题流程，按阶段运行（每个阶段可以重跑）。
#
#   bash shcm_bench.sh list        # 列出每首的编号和原文件夹
#   bash shcm_bench.sh skeleton    # 从俯拍视频提取手部骨架（MediaPipe，已提取的跳过）
#   bash shcm_bench.sh derive      # 导出成流程格式 + 自动指法（PMFinger v2 冻结配置）
#   bash shcm_bench.sh build       # 出题并渲染帧（和金标流程同一套规则；没有人工指法，所以不出指法题）
#
# 环境变量：
#   DAY     某一天的数据目录（默认 .../piano_zsy/data/SHCM_Piano/20261024）
#   WORK    工作目录（默认 ../shcm_work/<日期>）
#   CORNERS 俯拍画面里键盘四个角的像素坐标：A0 后沿、C8 后沿、C8 前沿、A0 前沿（默认是 10 月相机位置）
#   CAM     用哪路视频（默认 CAM_5.avi，俯拍）
#   MODEL   MediaPipe 手部模型（默认 PianoVAM-Code 自带的 hand_landmarker.task）
#   JOBS    骨架提取并行数（默认 8）；WORKERS 指法并行数（默认 8）；PER_SUBTYPE 每个子题题数（默认 20）
set -euo pipefail
cd "$(dirname "$0")"
G=/inspire/hdd/global_user/xuguangjun-253208580299
DAY=${DAY:-${G}/piano_zsy/data/SHCM_Piano/20261024}
WORK=${WORK:-../shcm_work/$(basename "${DAY}")}
CORNERS=${CORNERS:-"539,319 90,335 90,281 539,265"}
CAM=${CAM:-CAM_5.avi}
MODEL=${MODEL:-${G}/piano_zsy/code/PianoVAM-Code/FingeringDetection/hand_landmarker.task}
JOBS=${JOBS:-8}
WORKERS=${WORKERS:-8}
PY=${PY:-python3}
FREEZE="kernel=tri sigma_white=0.56 sigma_black=0.56 w_seq=0.5 conf_thresh=0.7 release_lateral=0.35 av_search=12"
step() { echo; echo "▶ $(date '+%H:%M:%S') $*"; }
mkdir -p "${WORK}/skeleton"

list() { ${PY} -B scripts/export_shcm.py --root "${DAY}" --skeleton "${WORK}/skeleton" --out "${WORK}/perf" --corners "${CORNERS}" --list; }

skeleton() {
  step "1 提取骨架 ${CAM} → ${WORK}/skeleton（并行 ${JOBS}）"
  list | while read -r tid dir; do
    [ -s "${WORK}/skeleton/${tid}.json" ] && continue
    [ -f "${dir}/${CAM}" ] || { echo "  ${tid} 没有 ${CAM}，跳过"; continue; }
    echo "${tid}	${dir}/${CAM}"
  done > "${WORK}/skeleton/todo.tsv"
  echo "  待提取 $(wc -l < "${WORK}/skeleton/todo.tsv") 首"
  while IFS=$'\t' read -r tid vid; do
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS}" ]; do sleep 1; done
    ( ${PY} -B scripts/extract_skeleton.py --video "${vid}" --model "${MODEL}" --out "${WORK}/skeleton/${tid}.json.tmp" 2>&1 \
        | grep -a "frames processed" | sed "s#^.*: #  ${tid}: #" \
      && mv "${WORK}/skeleton/${tid}.json.tmp" "${WORK}/skeleton/${tid}.json" ) &
  done < "${WORK}/skeleton/todo.tsv"
  wait
  echo "✅ 骨架：$(ls "${WORK}/skeleton"/*.json 2>/dev/null | wc -l) 首"
}

derive() {
  step "2a 导出 → ${WORK}/perf"
  ${PY} -B scripts/export_shcm.py --root "${DAY}" --skeleton "${WORK}/skeleton" --out "${WORK}/perf" \
    --corners "${CORNERS}" --cam "${CAM}" --ids-out "${WORK}/ids.txt"
  step "2b 指法与松键（冻结配置） → ${WORK}/fing"
  ${PY} -B scripts/run_fingering.py --inputs "${WORK}/perf/shcm-*[0-9].npz" --prior models/prior.npz --calib models/calib.npz \
    --out "${WORK}/fing" --workers "${WORKERS}" --set ${FREEZE}
  echo "✅ 第 2 步完成：$(wc -l < "${WORK}/ids.txt") 首"
}

build() {
  BENCH=${BENCH:-${WORK}/bench_v1}
  step "3 出题并渲染帧 → ${BENCH}"
  ${PY} -B scripts/build_bench_gold.py --work "${WORK}" --gold-ids "${WORK}/ids.txt" --out "${BENCH}" \
    --per-subtype "${PER_SUBTYPE:-20}"
  echo "✅ 第 3 步完成：${BENCH}/questions.jsonl，统计 ${BENCH}/report.md，浏览页 ${BENCH}/index.html"
}

case "${1:-}" in
  list) list ;;
  skeleton) skeleton ;;
  derive) derive ;;
  build) build ;;
  *) echo "用法: bash shcm_bench.sh list | skeleton | derive | build"; exit 2 ;;
esac
