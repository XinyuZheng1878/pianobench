#!/usr/bin/env bash
# 在启智平台（Linux）上跑 Benchmark 的步骤 1–7：装依赖 → 下载 PianoVAM → 出题 → 渲染。
# 和 mac_run.sh 做的事一样，代码和随机种子相同，所以生成的题目与 Mac 上 work_test 一致。
# 可以重复运行：已下载的文件、已渲染的片段会跳过。
#
#   cd pianobench && nohup bash platform_run.sh > /dev/null 2>&1 &     # 后台跑，关掉网页也不会停
#   tail -f logs/platform_run.log                                      # 看进度（Ctrl+C 只是退出查看，不会停止任务）
#
# 可选环境变量：
#   SPLITS=test         下载哪些 PianoVAM 划分（这些录音自带的人工指法也会下载）
#   WITH_GT=1           另外下载全部 11 段带人工指法的录音（只在评估指法准确率时需要，出题用不到）
#   DL_WORKERS=6        同时下载几个文件
#   BENCH_SPLIT=test    用哪个划分出题
#   PER_TASK=30         每个任务的事实数（先 30 试跑，正式约 600）
#   DATA=../pianovam_data   原始数据存放位置（默认放在仓库旁边，不进 git）
#   WORKERS=8           指法步骤的并行进程数
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
exec > >(tee -a logs/platform_run.log) 2>&1
die() { echo; echo "❌ $*"; echo "   完整日志：${PWD}/logs/platform_run.log"; exit 1; }
step() { echo; echo "▶ $(date '+%H:%M:%S') $*"; }
SPLITS=${SPLITS:-test}
BENCH_SPLIT=${BENCH_SPLIT:-test}
PER_TASK=${PER_TASK:-30}
DATA=${DATA:-../pianovam_data}
WORKERS=${WORKERS:-8}

step "1/5 检查环境"
PY=""
for c in python3.12 python3.11 python3.10 python3 python; do
  if command -v "${c}" >/dev/null 2>&1 && "${c}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then PY=$(command -v "${c}"); break; fi
done
[ -n "${PY}" ] || die "没找到 Python ≥ 3.10"
echo "Python：${PY}"
PKGS="numpy scipy opencv-python-headless pandas pillow requests huggingface_hub"
"${PY}" -c "import numpy, scipy, cv2, pandas, PIL, requests, huggingface_hub" 2>/dev/null || \
  "${PY}" -m pip install -q -U ${PKGS} || "${PY}" -m pip install -q -U -i https://pypi.tuna.tsinghua.edu.cn/simple ${PKGS} || die "pip 安装失败"
# ffmpeg：系统里没有就用 pip 装的静态版本（imageio-ffmpeg 自带 libx264）
if ! command -v ffmpeg >/dev/null; then
  "${PY}" -c "import imageio_ffmpeg" 2>/dev/null || "${PY}" -m pip install -q imageio-ffmpeg || \
    "${PY}" -m pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple imageio-ffmpeg || die "装不上 ffmpeg"
  mkdir -p .bin && ln -sf "$("${PY}" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" .bin/ffmpeg
  export PATH="${PWD}/.bin:${PATH}"
fi
ffmpeg -version | head -1
"${PY}" -B scripts/selftest.py | tail -1 | grep -q "ALL PASS" || die "自检没通过"
echo "环境正常"

step "2/5 下载 PianoVAM（${SPLITS} 划分）到 ${DATA}"
# 平台直连 HuggingFace 经常超时或很慢，默认走国内镜像（想直连就设 HF_ENDPOINT=https://huggingface.co）
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
# 新的 Xet 下载通道在平台上会报 “error decoding response body”，关掉改用普通下载
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
echo "HuggingFace 地址：${HF_ENDPOINT:-https://huggingface.co}"
GT_FLAG=$([ "${WITH_GT:-0}" = 1 ] || echo --no-gt)
"${PY}" scripts/download_pianovam.py --root "${DATA}" --splits "${SPLITS}" ${GT_FLAG} --dry-run | head -1 || die "连不上数据集"
export HF_HUB_DOWNLOAD_TIMEOUT=60
ok=0
for try in 1 2 3 4 5 6; do
  if "${PY}" scripts/download_pianovam.py --root "${DATA}" --splits "${SPLITS}" ${GT_FLAG} --workers "${DL_WORKERS:-6}"; then ok=1; break; fi
  echo "下载出错，第 ${try} 次重试（已下载的文件会跳过）……"; sleep 5
done
[ "${ok}" = 1 ] || die "下载多次失败；检查网络后重新运行本脚本，会接着下"

step "3/5 选出 ${BENCH_SPLIT} 划分里已下载的录音"
RECS=$("${PY}" - "${DATA}" "${BENCH_SPLIT}" <<'PY'
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
DATA="${DATA}" RECS="${RECS}" PER_TASK="${PER_TASK}" WORK="work_${BENCH_SPLIT}" WORKERS="${WORKERS}" PY="${PY}" \
  bash run_all.sh || die "流程出错，见上方输出"

step "5/5 完成"
echo "题目：${PWD}/work_${BENCH_SPLIT}/release/questions.jsonl"
echo "事实卡：${PWD}/work_${BENCH_SPLIT}/cards.jsonl"
echo "复核清单：${PWD}/work_${BENCH_SPLIT}/bench/review.tsv，复核图：${PWD}/work_${BENCH_SPLIT}/release/review/"
echo "出题统计：${PWD}/work_${BENCH_SPLIT}/bench/gen_report.txt"
echo "✅ 全部完成"
