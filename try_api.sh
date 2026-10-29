#!/usr/bin/env bash
# 用真实 API 试跑扩写：每个子类型抽几张事实卡 → 扩写 → 质检打分 → 拼成题目 → 生成可读的报告 cases.md
#
#   export OPENAI_API_KEY=你的key          # 只放在环境变量里，不要写进文件
#   BASE_URL=<兼容接口地址> MODEL=<模型名> bash try_api.sh
#
# 可选：SAMPLE=1（每个子类型抽几张，默认 1，约 25 张卡）  JUDGE_MODEL=（质检用的模型，默认同 MODEL）  WORK=work_test
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
PY=$(command -v python || command -v python3)
: "${BASE_URL:?请设置 BASE_URL}"; : "${MODEL:?请设置 MODEL}"; : "${OPENAI_API_KEY:?请先 export OPENAI_API_KEY=你的key}"
W=${WORK:-work_test}
OUT="${W}/try_$(date +%m%d_%H%M)"
mkdir -p "${OUT}"
echo "▶ 1/4 扩写（每个子类型 ${SAMPLE:-1} 张卡，模型 ${MODEL}）"
${PY} -B scripts/expand_qa.py run --cards "${W}/cards.jsonl" --release "${W}/release" --out "${OUT}/expansions.jsonl" \
  --base-url "${BASE_URL}" --model "${MODEL}" --sample "${SAMPLE:-1}" --workers 2
echo "▶ 2/4 质检打分（模型 ${JUDGE_MODEL:-$MODEL}）"
${PY} -B scripts/expand_qa.py judge --cards "${W}/cards.jsonl" --expansions "${OUT}/expansions.jsonl" \
  --out "${OUT}/judgments.jsonl" --base-url "${BASE_URL}" --model "${JUDGE_MODEL:-$MODEL}" --workers 2
echo "▶ 3/4 拼成最终题目"
${PY} -B scripts/expand_qa.py apply --release "${W}/release" --cards "${W}/cards.jsonl" \
  --expansions "${OUT}/expansions.jsonl" --judgments "${OUT}/judgments.jsonl" --out "${OUT}/questions_nl.jsonl"
echo "▶ 4/4 生成报告"
${PY} -B scripts/expand_qa.py report --questions "${OUT}/questions_nl.jsonl" --out "${OUT}/cases.md"
echo "✅ 报告：${PWD}/${OUT}/cases.md"
command -v open >/dev/null && open "${OUT}/cases.md" || true
