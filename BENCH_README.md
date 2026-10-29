# 钢琴手部视频问答 Benchmark：构建与评测代码

在 PMFinger v2 上新增的 Benchmark 部分。原有的 `pmfinger/` 指法与松键代码没有改动。所有设计都对应文档《钢琴手部视频问答 Benchmark 与构建流程》里的 Benchmark 页和 Pipeline 页。

## 一条命令跑完

```bash
DATA=/path/to/PianoVAM_v1 bash run_all.sh                       # 步骤 1–7
BASE_URL=http://localhost:8001/v1 MODEL=Qwen/Qwen3-VL-8B-Instruct TAG=qwen bash run_all.sh   # 加上步骤 8
```

每一步也能单独运行，命令写在 `run_all.sh` 里。依赖：numpy、scipy、opencv-python、pandas、pillow、requests，以及 ffmpeg。

## 八个步骤与对应文件

| 步骤 | 脚本 | 输出 | 新做 / 已有 |
|---|---|---|---|
| 1 数据导出 | `scripts/export_pianovam.py` | `perf/<rec>.npz`（schema 格式，另带真实松键 `note_key_offset` 和人工金标 `gt_hand/gt_finger`）、`<rec>.skeleton.npz`（21 点骨架）、`<rec>.meta.json` | 新做：服务器导出脚本的本地版，只支持 PianoVAM |
| 2 质检 | `scripts/check_octave.py` | `qc_octave.json` | 已有 |
| 3 指法与松键 | `scripts/run_fingering.py` | `fing/<rec>.fing.npz` | 已有，冻结配置；`av_search` 放宽到 ±12 帧 |
| 4 状态派生 | `pmfinger/states.py`、`scripts/check_states.py` | 运动状态、手的状态、乐理标注 | 新做 |
| 5 出题 | `pmfinger/bench.py`、`scripts/gen_bench.py` | `bench/questions.jsonl`、`facts.jsonl`、`review.tsv` | 新做 |
| 6 人工复核 | `render_clips.py --review` | `release/review/<fact_id>.jpg` 复核图 | 新做 |
| 7 渲染与发布 | `scripts/render_clips.py` | `release/clips/<clip_id>/`：video.mp4、frames_8fps/15fps/2fps/4fps、handskeleton.npz、keys.json、pose.json、notes.json、clip.mid | 新做 |
| 8 推理与评分 | `scripts/run_models.py`、`scripts/evaluate.py` | `runs/*.jsonl`、`report/report.md` | 由 run_infer.py、score.py 改写 |

## 题目可信度的几道关

- **指法**：可信 = 有人工金标（gold），或者 status 为 ok、置信度 ≥ 0.9、并且与 PianoVAM 自动标签一致（silver）。依赖指法的题，区间内涉及的每个音都必须可信。
- **松键**：PianoVAM 直接用钢琴记录的松键时间。
- **区间边界**：区间两端前后 150 ms 内，这只手没有按键或松键。
- **答案分档**：真值落在分档线附近的不出题，例如保持时长、跨度、移动距离。
- **骨架跟踪**：片段内 95% 以上的帧要有骨架。
- **复核清单**：依赖指法或松键的事实写进 `review.tsv`，按涉及的音数降序、最低置信度升序排列，越靠前越该优先复核。

## 在 GPU 服务器上跑基线

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001 --max-model-len 65536 --limit-mm-per-prompt '{"image": 90}'
python -B scripts/run_models.py --release work/release --base-url http://localhost:8001/v1 \
    --model Qwen/Qwen3-VL-8B-Instruct --tag qwen --setting F --out work/runs/qwen_F.jsonl
python -B scripts/run_models.py ... --setting blind --out work/runs/qwen_blind.jsonl
python -B scripts/evaluate.py --questions work/release/questions.jsonl --runs work/runs/*.jsonl --out work/report
```

JoyAI-VL-Interaction 的权重是标准的 Qwen3-VL 结构，同样用 `vllm serve jdopensource/JoyAI-VL-Interaction` 起服务。

设置 F 一道题最多 80 帧，按长边 1280 算，单题大约 3–5 万个视觉 token，需要把 `--max-model-len` 开到 64k 左右。显存不够时：
- 加 `--max-side 896` 缩小帧；
- 或者用 4 fps 的消融帧，把题目里的 `frame_fps` 改成 4。
