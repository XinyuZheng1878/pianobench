#!/usr/bin/env bash
# 在自己的 Mac 上跑 Benchmark 的步骤 1–7（下载数据 → 出题 → 渲染）。可以重复运行，已完成的部分会跳过。
#
#   cd ~/projects/pianobench && bash mac_run.sh
#
# 可选环境变量：
#   SPLITS=test,valid   下载哪些 PianoVAM 划分（test 出正式测试题，valid 以后做开发集）
#   BENCH_SPLIT=test    用哪个划分出题
#   PER_TASK=30         每个任务的事实数（先 30 试跑，正式约 600）
#   DATA=...            PianoVAM 存放位置（默认复用 ~/projects/pianovam_qa/data/pianovam，已下好的 5 段不会重下）
#   YES=1               下载前不再询问
set -uo pipefail
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"
if [ -z "${_CAFF:-}" ] && command -v caffeinate >/dev/null; then exec env _CAFF=1 caffeinate -i bash "$0" "$@"; fi
mkdir -p logs
exec > >(tee -a logs/mac_run.log) 2>&1
die() { echo; echo "❌ $*"; echo "   完整日志：${PWD}/logs/mac_run.log"; exit 1; }
step() { echo; echo "▶ $*"; }
SPLITS=${SPLITS:-test,valid}
BENCH_SPLIT=${BENCH_SPLIT:-test}
PER_TASK=${PER_TASK:-30}
DATA=${DATA:-../pianovam_qa/data/pianovam}

step "1/5 检查环境"
command -v ffmpeg >/dev/null || { command -v brew >/dev/null && brew install ffmpeg || die "缺少 ffmpeg：先装 Homebrew，再 brew install ffmpeg"; }
PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "${c}" >/dev/null 2>&1 && "${c}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then PY=$(command -v "${c}"); break; fi
done
[ -n "${PY}" ] || die "没找到 Python ≥ 3.10，可以 brew install python@3.11"
[ -x .venv/bin/python ] || "${PY}" -m venv .venv || die "创建虚拟环境失败"
source .venv/bin/activate
PKGS="numpy scipy opencv-python-headless pandas pillow requests huggingface_hub"
python -c "import numpy, scipy, cv2, pandas, PIL, requests, huggingface_hub" 2>/dev/null || \
  python -m pip install -q -U ${PKGS} || python -m pip install -q -U -i https://pypi.tuna.tsinghua.edu.cn/simple ${PKGS} || die "pip 安装失败"
python -B scripts/selftest.py | tail -1 | grep -q "ALL PASS" || die "自检没通过"
echo "环境正常"

step "2/5 下载 PianoVAM（${SPLITS} 划分 + 带人工指法的录音）到 ${DATA}"
if [ -z "${HF_ENDPOINT:-}" ] && ! curl -s -o /dev/null --max-time 8 https://huggingface.co; then export HF_ENDPOINT=https://hf-mirror.com; fi
echo "HuggingFace 地址：${HF_ENDPOINT:-https://huggingface.co}"
python scripts/download_pianovam.py --root "${DATA}" --splits "${SPLITS}" --dry-run | head -1 || die "连不上数据集"
if [ "${YES:-0}" != 1 ]; then read -r -p "按上面的大小下载？已下载的文件会跳过 [y/N] " ans; [[ "${ans}" =~ ^[yY] ]] || die "已取消"; fi
export HF_HUB_DOWNLOAD_TIMEOUT=60   # 某个文件 60 秒没有进展就断开重试，而不是一直卡住
ok=0
for try in 1 2 3 4 5 6; do
  if python scripts/download_pianovam.py --root "${DATA}" --splits "${SPLITS}"; then ok=1; break; fi
  echo "下载出错，第 ${try} 次重试（已下载的文件会跳过）……"; sleep 5
done
[ "${ok}" = 1 ] || die "下载多次失败；检查网络后重新运行本脚本，会接着下"

step "3/5 选出 ${BENCH_SPLIT} 划分里已下载的录音"
RECS=$(python - "${DATA}" "${BENCH_SPLIT}" <<'PY'
import json, os, sys
root, split = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(root, "metadata.json"), encoding="utf-8"))
recs = sorted(e["record_time"] for e in m.values() if e.get("split") == split
              and all(os.path.exists(os.path.join(root, d, e["record_time"] + x))
                      for d, x in (("Video", ".mp4"), ("Handskeleton", ".json"), ("Fingering", ".tsv"))))
print(" ".join(recs))
PY
)
[ -n "${RECS}" ] || die "${BENCH_SPLIT} 划分里没有下载完整的录音"
echo "共 $(echo ${RECS} | wc -w | tr -d ' ') 段：${RECS}"

step "4/5 运行步骤 1–7（每任务 ${PER_TASK} 个事实）"
DATA="${DATA}" RECS="${RECS}" PER_TASK="${PER_TASK}" WORK="work_${BENCH_SPLIT}" WORKERS=4 PY=python bash run_all.sh || die "流程出错，见上方输出"

step "5/5 完成"
echo "题目：${PWD}/work_${BENCH_SPLIT}/release/questions.jsonl"
echo "复核清单：${PWD}/work_${BENCH_SPLIT}/bench/review.tsv，复核图：${PWD}/work_${BENCH_SPLIT}/release/review/"
echo "出题统计：${PWD}/work_${BENCH_SPLIT}/bench/gen_report.txt"
