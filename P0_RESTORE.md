# P0 恢复：独立的离线验收版本

基于 `codex/vam-batch` 的 `61ca1f1`，上游基线 `37169e5`。分支 `codex/p0-restore`，对应 issue https://github.com/XinyuZheng1878/pianobench/issues/1 。本步骤不调用模型 API，不修改在跑的 release_v2，不更改历史题答案，不 push。

## 入口和数据流

`scripts/p0_restore.py` 是此次唯一新出题入口：从冻结的 `release_v2/manifest.json` 选恰好两段 dev，交叉拒绝 gold_ids；只读已有 perf/fing/skeleton/meta，不重新推断标签。`pmfinger/p0_states.load_raw` 对齐 MIDI、A/V 偏移和原视频帧数，拒绝非零 frame_offset；读取未经插值或居中平滑的21点坐标，重建掌心和指尖。`p0_rules.Gen` 是从 bench 迁入的规则模块，接入欣宇的导出文件、键盘几何和现有帧渲染，不再在 bench 下另起生产流水线。

`scan` 每0.5秒滑动扫描，按视频/子类型各最多2题（参数可改），同语义窗口重叠去重。统计 scanned/eligible/selected/rejected；没有合格事实就是0，不放松门槛凑样例。下一键选择/填空同一生成调用配对。当前按固定扫描顺序选取首批合格窗口，因此时间位置和答案分布还需要后续规模试点衡量。

所有输出放在全新 out：`results/<qid>.json`、`counts/<video>.json`、`dry_run.json`、`media/<qid>/manifest.json`、JPEG帧和帧包、`index.html`。out 不得是 source 的父/子目录，已有输出也拒绝覆盖。server 中 out/perf 为只读读取用途的源缓存链接；渲染不写源缓存。

## 规则与语义

- 手：独立人工手标签优先；否则 PMFinger 判手置信度≥0.9且必须与独立自动手标签一致。自动标签缺失不视为一致。
- 手指：只有人工 GT 手+手指组合有效，含义只限 finger_at_onset。所有指法题进入 needs_human_GT_review；没有 GT 不生成题。
- 人审：批准记录必须同时匹配 qid、fact_hash、decision=approve、非空 reviewer。`verify_frozen` 重算公有问题、私有目标、证据和源文件哈希，不能篡改答案/证据或关闭人审标志后沿用批准。`compose_sft` 消费此批准检查；不是只写一个没人读取的 review 文件。
- 掌心用0/5/9/13/17均值；跨度用指定交付帧拇指尖与小指尖横距。T3跨度±0.1秒同档、变动≤0.5白键、距4/6/8边界≥0.3；位移端点吸附实际交付帧，±50ms中位数，原端点标签一致，距1/4/8边界≥0.4。
- 几何缺失拒绝，不插值；掌心及所有指尖相邻帧跳变>1白键拒绝。T4横移进3出1.5白键/秒，0.25–1.2秒，≥7白键；吸附后的端点必须继续满足门槛。T5当前位置仅用切点前0.1秒，未来位置仅用于私有标签，方向阈值±1、余量0.3。
- 完整时间线先做锚定40ms非链式分组，再取窗口/手指子集。切断完整组时拒绝；press_visible要求可见的未按→按下，countable还检查不同组及任何同键重按之间的可见松开。时长可观测上下界必须整体落入同一档。
- 新门槛也接入 `batch_sft.evidence`，覆盖所有当前启用核心题型及T5历史证据；这是本机新分支代码，未同步到运行中的服务器生产 code。
- T5最后交付帧就是“现在”；未来状态查询该键全部按下/松开，计入重按；下一组完整且单音、起点在0.5秒内。四个唯一选项均在当前掌心±5白键范围，重复上一键可以是真值，上一键也可按50%概率作为干扰项。
- 预测的答案、未来事件/位置/难度标签只放 private_target；`writer_input` 只返回冻结问题和截止末帧的原子。键选项是公共候选集合，真值索引保留在 private_target。程序填写最终 X，逐句校验完整的时间/手/指/键/动作关系。
- 同名不同语义采用新的 semantic_id：新40ms组数为 `vam-p0-restore-v1/onset_group_count_40ms`，不改旧核心单键onset计数；新三音序列用 `third_onset_interval`，不改旧T4旋律首onset→末note-off答案。新掌心/跨度语义也以新profile隔离，历史 `pmfinger/bench.py` 生成函数保留。
- 五态、穿指/跨指、C1踏板未恢复。

## 校验与下一阶段接口

本步只产离线候选，不自动把候选当SFT通过样本。`writer_input`、`atom_sentence`、`compose_sft` 是下步B/D适配必须复用的入口。观察使用固定受控句；analysis逐句等于所引原子的规范关系句；最终格式为 `【观察】…【分析】…所以答案是{X}`，选择题X只有字母。指法未人审批准不能合成SFT。

`scripts/p0_api.py` 已接入独立B/D/judge调度，绑定完整提示、代码和媒体哈希缓存，事实变化拒绝沿用旧缓存。指法先读回匹配fact_hash的人工批准，无批准时0次模型调用；批准后的结果标为human_accepted。其他通过样本可auto_accepted，低分/硬错分流needs_review。它只接受dev候选，输出独立sft_dev，不能指向生产目录。不得直接把新题塞入旧核心 `program_summary`（它的T3分支只认识单键计数）。本步仅以离线假API测试控制流，未执行真实API入口，不宣称模型盲审、judge或人工验收已通过。

## 离线验证

本机24项原生产回归、16项P0测试方法通过，其中包含22个恢复子类型各自正例+反例的参数化检查；服务器同两套及上游 `scripts/selftest.py` 全通过。附加反例覆盖：GT手无GT指、auto缺失、分组起点被过滤、预测窗/切点截断和弦、同键重按、短按不可见、时长跨档、手指未知竞争者、首次事件不能顺延、未来骨架变化不改变历史原子、过期/篡改人审批准备份。

实数据：只用 `2024-02-17_21-37-57`、`2024-03-13_11-32-42` 两段非金标dev。服务器路径 `users/qihengwang/pianobench_runs/p0_restore_20261030/dryrun_v2`。39条候选、39条成功交付、0次API；指法为0，部分严格子类型无真实合格样例。逐子类型扫描/淘汰数量以 dry_run.json 为准。所有39条在最终代码下重算fact_hash通过。

服务器只从生产目录读缓存，代码副本和输出均另建；PID265746保留。完整实跑记录和样题页在本机 bench/runs/p0_restore_20261030 下，训练前仍需人工核验音画偏移/可见性及独立API试点确认。
