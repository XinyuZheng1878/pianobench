#!/usr/bin/env bash
# 重新出题 → 渲染（已渲染的片段会跳过）→ 生成给大模型的事实卡 → 用模拟模型把扩写流程试跑一遍。
#   cd ~/projects/pianobench && bash mac_cards.sh
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
W=${WORK:-work_test}
PER_TASK=${PER_TASK:-30}
echo "▶ 1/4 出题"
python -B scripts/gen_bench.py --perf "${W}/perf/[0-9]*[0-9].npz" --fing "${W}/fing" --out "${W}/bench" --per-task "${PER_TASK}" | tail -2
echo "▶ 2/4 渲染（已有的视频和帧会跳过，只更新标注和复核图）"
python -B scripts/render_clips.py --questions "${W}/bench/questions.jsonl" --perf "${W}/perf" --fing "${W}/fing" \
  --out "${W}/release" --ablation-fps 2 4 --review | tail -1
echo "▶ 3/4 生成事实卡"
python -B scripts/make_cards.py --bench "${W}/bench" --release "${W}/release" --perf "${W}/perf" --fing "${W}/fing" --out "${W}/cards.jsonl"
echo "▶ 4/4 用模拟模型试跑扩写和自动校验（不调用任何 API）"
rm -f "${W}/expansions_mock.jsonl"
python -B scripts/expand_qa.py run --cards "${W}/cards.jsonl" --release "${W}/release" --out "${W}/expansions_mock.jsonl" --mock --workers 1 | tail -1
python -B scripts/expand_qa.py apply --release "${W}/release" --cards "${W}/cards.jsonl" --expansions "${W}/expansions_mock.jsonl" --out "${W}/release/questions_nl_mock.jsonl"
python -B scripts/expand_qa.py run --cards "${W}/cards.jsonl" --release "${W}/release" --out /dev/null --show 1 | tail -3
echo "✅ 完成：事实卡 ${PWD}/${W}/cards.jsonl"
