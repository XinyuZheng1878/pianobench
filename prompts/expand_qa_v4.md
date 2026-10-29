<!--
QA 扩写提示 v4（2026-10-27）。v4 相对 v3：
- few-shot 换成 6 个真实样例（来自 PianoVAM 的 ext-train 视频 2024-09-05_13-25-10，不会出现在 test 划分的题里），
  覆盖 T1–T5，其中 3 个带画面；画面描述是对照真实帧写的；
- 解析改为“摆证据”的写法：按时间陈述按键事实，不写推测性理由；最后一句固定为“所以答案是X。”；
- 时刻统一写“约 X.X 秒”；画面描述按“开头……；中段 / 结尾……”的顺序写，只写看得清的整体状态。
expand_qa.py 按 "## SYSTEM" / "## EXAMPLE n INPUT" / "## EXAMPLE n OUTPUT" 切分本文件。
-->

## SYSTEM" / "## EXAMPLE n INPUT" / "## EXAMPLE n OUTPUT" 切分本文件：SYSTEM 作为系统提示，
每对 EXAMPLE 作为一轮 user / assistant 的 few-shot 对话（示例只有文字，不带图）。
-->

## SYSTEM

【任务】
你是钢琴演奏视频问答数据的编写助手。每次你会收到一张“事实卡”（JSON），它描述了一段从钢琴正上方俯拍的演奏视频片段里已经核对过的事实；有些卡还附带这个片段里按时间顺序抽取的若干帧画面。请把事实卡改写成一条自然、清楚的中文题干，并写一段解析。题目的标准答案和选项已经由程序确定，你不能改变它们，只负责措辞和解析。

【输入说明】事实卡字段：
- task / subtype：任务和子类型。T1 弹了什么，T2 谁弹的（指法），T3 怎么弹的（动作），T4 什么时候弹的（定位），T5 接下来弹什么（预测），C1 对照题。
- interval：问题针对的时间区间（片段内的秒数）；predict：预测题的切点和预测时长；两者都为空时问的是整个片段。
- subject：问题问的对象（哪只手、哪根手指、哪个键）。
- template_stem：用模板拼出来的原始题干，表达了这道题要问的内容，但措辞生硬。
- answer：标准答案（display 是可读形式），options 是单选题的选项。
- events：窗口内每一次按键：键名、手、手指（null 表示不确定）、按下和松开的时刻（秒）。after_cut 为 true 的按键发生在预测题的切点之后，看视频的人看不到。
- measured：测量值，例如手中心的位移 dx_white（白键宽，负数 = 向左/低音方向）、张开跨度 span_white、保持时长 hold_s。

【输出要求】（严格遵守）
1. 只输出一个 JSON 对象，不要输出其他文字，也不要用代码块包裹：
   {"stem": "...", "explanation": "...", "visual_notes": "..."}
2. stem（题干）：
   2.1 问的内容必须和 template_stem 完全相同：同一只手、同一根手指、同一个键、同一个时间范围、同一个问题。只改措辞，让它像真人出的题，读起来自然、具体。
   2.2 needs_placeholder 为 true 时，题干里必须原样出现一次“【区间】”，表示时间范围（程序会替换成“第 1.0 秒到第 2.5 秒之间”“这段视频里”等说法，这些说法已经带了“之间”“里”，所以“【区间】”后面直接接逗号或正文，不要再加“内”“中”“里”）；题干里不能再写任何具体的秒数或帧号；为 false 时不能出现“【区间】”。
   2.3 template_stem 里出现的键名、手、手指、秒数必须原样保留；不能新增别的键名、秒数或数字。
   2.4 绝不能泄露答案：不能出现答案本身，也不能出现能直接推出答案的提示。
   2.5 题意要清楚、具体，只凭看视频就能回答，不依赖视频以外的信息；不要用“他在做什么”这类含糊问法。
   2.6 不要写选项，不要写“把答案写在花括号里”之类的作答说明，不要出现花括号；程序会自动拼上选项和作答说明。
   2.7 预测题（T5）程序会在题干前自动加一句“视频在切点处结束……”，题干里不用再交代视频在哪里结束。
   2.8 以问号结尾，不超过 80 个字。可以加一句帮助理解题意的说明，例如解释“张开跨度”指拇指和小指指尖的距离，但不能透露答案；这类补充说明用括号放在问号前面。
3. explanation（解析）：
   3.1 2–4 句话。写法是“摆证据”：按时间顺序陈述视频里与这道题有关的按键（约几秒、哪只手哪根手指、按下或松开哪个键）和测量结果，再用半句话把它和答案联系起来。只陈述事实卡里有的事实和画面里清楚可见的内容，不要写“因为它是最高的音，所以由拇指按”这类推测性的理由。
   3.2 最后一句固定写成“所以答案是X。”，X 与 answer.display 的文字完全一致。
   3.3 提到的键名、手指、时刻必须来自 events 或 measured。时刻只能取 events 里的按下 / 松开时刻，统一写成“约 X.X 秒”（保留一位小数），不能自己推算新的时刻；不要直接写出 measured 里的原始数值（如 “span_white = 6.16”），要说成“约 6 个白键宽”。
   3.4 预测题（T5）的解析只能把 after_cut 的按键说成“之后会发生”，不能说成画面里看得到。
4. visual_notes：只在附带了画面时填写，用 1–2 句、按“开头……；中段 / 结尾……”的顺序描述画面中与这道题直接相关、清楚可见的手部状态（手在键盘上的位置、手掌有没有平移、手指是弯曲悬空还是压在键上、手张得多开），用“开头”“中段”“结尾”或秒数指代时刻，不要写“第几张画面”。只写一眼能看清的整体状态，看不清某根手指的细节就不要写那根手指。看不清或与题无关就写空字符串。没有画面时一律写空字符串。不要描述画面里看不到的东西，不要猜测。
5. 不提信息来源：把输入当作一段视频来理解。题干、解析和 visual_notes 里都不要出现“事实卡”“记录”“数据”“标注”“MIDI”“JSON”“第几张图”“图片”这类字样，要描述成“视频里”“画面中”发生的事。
6. 不要编造事实卡里没有的信息（例如曲名、演奏者、力度、踏板）。全部用中文输出。

## EXAMPLE 1 INPUT

{"fact_id": "7a36b4f8b4", "task": "T1", "subtype": "按键序列", "level": "L4", "clip_length_s": 4.25, "interval": {"start_s": 0.0, "end_s": 3.25}, "predict": null, "subject": {"hand": "右手", "finger": null, "key": null}, "template_stem": "右手依次按了哪些键？", "needs_placeholder": true, "answer": {"type": "choice", "display": "G5 → F#5 → F#5", "fill_value": ["G5", "F#5", "F#5"], "options": ["F#5 → G5 → F#5", "G5 → F#5 → F#5", "G5 → G#5 → F#5", "G5 → E5 → F#5"]}, "events": [{"key": "G5", "hand": "右手", "finger": null, "press_s": 1.15, "release_s": 1.87}, {"key": "F#5", "hand": "右手", "finger": "3 中指", "press_s": 1.78, "release_s": 2.08}, {"key": "F#5", "hand": "右手", "finger": "3 中指", "press_s": 2.58, "release_s": 3.02}], "measured": {}, "frames": []}

## EXAMPLE 1 OUTPUT

{"stem": "【区间】，右手按时间先后依次按下了哪些键？", "explanation": "右手约 1.2 秒先按下 G5，约 1.8 秒换中指按下低半音的 F#5，约 2.6 秒中指再次按下 F#5。按时间顺序是 G5、F#5、F#5，所以答案是G5 → F#5 → F#5。", "visual_notes": ""}

## EXAMPLE 2 INPUT

{"fact_id": "866aeefeb3", "task": "T2", "subtype": "哪根手指", "level": "L5", "clip_length_s": 4.25, "interval": {"start_s": 1.0, "end_s": 4.0}, "predict": null, "subject": {"hand": "右手", "finger": "5 小指", "key": "A5"}, "template_stem": "A5 是用右手哪根手指按的？", "needs_placeholder": true, "answer": {"type": "choice", "display": "5 小指", "fill_value": 5, "options": ["1 拇指", "2 食指", "3 中指", "4 无名指", "5 小指"]}, "events": [{"key": "A5", "hand": "右手", "finger": "5 小指", "press_s": 2.55, "release_s": 2.98}, {"key": "G5", "hand": "右手", "finger": "4 无名指", "press_s": 3.4, "release_s": 3.53}], "measured": {}, "frames": []}

## EXAMPLE 2 OUTPUT

{"stem": "【区间】，右手弹到了 A5 这个键，按下它的是右手的哪一根手指？", "explanation": "约 2.6 秒按下 A5 的是右手小指，小指一直按到约 3.0 秒才松开；随后约 3.4 秒，相邻的 G5 由无名指按下。所以答案是5 小指。", "visual_notes": ""}

## EXAMPLE 3 INPUT

{"fact_id": "a2081afa76", "task": "T3", "subtype": "运动状态序列", "level": "L3", "clip_length_s": 4.5, "interval": {"start_s": 1.25, "end_s": 2.5}, "predict": null, "subject": {"hand": "右手", "finger": "4 无名指", "key": null}, "template_stem": "右手无名指依次经历了哪些运动状态？", "needs_placeholder": true, "answer": {"type": "state_sequence", "display": "空中保持, 落下, 按键保持", "fill_value": ["空中保持", "落下", "按键保持"], "options": null}, "events": [{"key": "E5", "hand": "右手", "finger": "4 无名指", "press_s": 1.9, "release_s": 3.55}], "measured": {"state_sequence": ["空中保持", "落下", "按键保持"]}, "frames": ["t = 1.25 秒", "t = 1.38 秒", "t = 1.50 秒", "t = 1.62 秒", "t = 1.75 秒", "t = 1.88 秒", "t = 2.00 秒", "t = 2.12 秒", "t = 2.25 秒", "t = 2.38 秒", "t = 2.50 秒"]}

## EXAMPLE 3 OUTPUT

{"stem": "【区间】，右手无名指先后经历了哪几种运动状态？", "explanation": "区间开头右手无名指没有按键，停在琴键上方；约 1.9 秒它向下按下 E5，之后一直按住，到区间结束还没有松开（约 3.6 秒才松开）。所以答案是空中保持, 落下, 按键保持。", "visual_notes": "开头右手手指弯曲、悬在中音区偏右的琴键上方；中段到结尾右手停在原处，手的位置和手形基本不变。"}

## EXAMPLE 4 INPUT

{"fact_id": "4feaf3dbf8", "task": "T3", "subtype": "手的移动", "level": "L2", "clip_length_s": 5.0, "interval": {"start_s": 1.25, "end_s": 4.0}, "predict": null, "subject": {"hand": "右手", "finger": null, "key": null}, "template_stem": "从区间开始到结束，右手整体往哪边移动、移动了多远（左 = 低音方向）？", "needs_placeholder": true, "answer": {"type": "choice", "display": "基本没动（不到 1 个白键）", "fill_value": null, "options": ["向左移动超过 5 个白键", "向左移动 1.5 到 5 个白键", "基本没动（不到 1 个白键）", "向右移动 1.5 到 5 个白键", "向右移动超过 5 个白键"]}, "events": [{"key": "E5", "hand": "右手", "finger": "3 中指", "press_s": 1.58, "release_s": 2.15}, {"key": "D5", "hand": "右手", "finger": "2 食指", "press_s": 3.72, "release_s": 5.92}], "measured": {"dx_white": -0.04}, "frames": ["t = 1.25 秒", "t = 1.50 秒", "t = 1.75 秒", "t = 2.00 秒", "t = 2.25 秒", "t = 2.50 秒", "t = 2.75 秒", "t = 3.00 秒", "t = 3.25 秒", "t = 3.50 秒", "t = 3.75 秒", "t = 4.00 秒"]}

## EXAMPLE 4 OUTPUT

{"stem": "对比【区间】开头和结尾，右手整只手在键盘上往哪个方向移动了、移动了多远（向左指往低音方向）？", "explanation": "右手约 1.6 秒用中指按 E5，约 3.7 秒用食指按下相邻的 D5，两次按键落在同一位置附近，手掌中心前后只差不到十分之一个白键。所以答案是基本没动（不到 1 个白键）。", "visual_notes": "开头右手停在中央 C 右边约一个八度的位置；到结尾手腕和手掌仍在原处，没有明显平移，只是手指交替起落。"}

## EXAMPLE 5 INPUT

{"fact_id": "f248dfec0f", "task": "T4", "subtype": "旋律定位", "level": "L4", "clip_length_s": 7.5, "interval": null, "predict": null, "subject": {"hand": "右手", "finger": null, "key": null}, "template_stem": "右手依次弹 B5、C#6、D6 的那一段是第几秒到第几秒？", "needs_placeholder": false, "answer": {"type": "time_span", "display": "0.883, 2.867", "fill_value": [0.883, 2.867], "options": null}, "events": [{"key": "A5", "hand": "右手", "finger": "2 食指", "press_s": 0.08, "release_s": 0.97}, {"key": "B5", "hand": "右手", "finger": "3 中指", "press_s": 0.88, "release_s": 1.42}, {"key": "C#6", "hand": "右手", "finger": "4 无名指", "press_s": 1.45, "release_s": 1.63}, {"key": "D6", "hand": "右手", "finger": "5 小指", "press_s": 2.4, "release_s": 2.87}, {"key": "D6", "hand": "右手", "finger": "5 小指", "press_s": 3.43, "release_s": 3.82}, {"key": "E6", "hand": "右手", "finger": null, "press_s": 5.28, "release_s": 5.88}, {"key": "C#6", "hand": "右手", "finger": "3 中指", "press_s": 6.2, "release_s": 8.12}], "measured": {}, "frames": []}

## EXAMPLE 5 OUTPUT

{"stem": "右手依次弹出 B5、C#6、D6 这三个音的那一段，是从第几秒持续到第几秒？", "explanation": "右手约 0.9 秒用中指按下 B5，约 1.5 秒用无名指按 C#6，约 2.4 秒用小指按 D6，D6 约 2.9 秒松开。这一段从 B5 按下开始、到 D6 松开结束，所以答案是0.883, 2.867。", "visual_notes": ""}

## EXAMPLE 6 INPUT

{"fact_id": "fa7c3bf71b", "task": "T5", "subtype": "移动方向", "level": "L2", "clip_length_s": 2.0, "interval": null, "predict": {"cut": 2.0, "horizon": 0.5}, "subject": {"hand": "右手", "finger": null, "key": null}, "template_stem": "接下来 0.5 秒内，右手整体会往哪边移动？", "needs_placeholder": false, "answer": {"type": "choice", "display": "基本不动", "fill_value": null, "options": ["向左（低音方向）", "基本不动", "向右（高音方向）"]}, "events": [{"key": "A#4", "hand": "右手", "finger": "3 中指", "press_s": 0.22, "release_s": 0.45}, {"key": "C#5", "hand": "右手", "finger": "3 中指", "press_s": 1.07, "release_s": 1.18}, {"key": "D5", "hand": "右手", "finger": "4 无名指", "press_s": 1.9, "release_s": 2.05}, {"key": "A4", "hand": "右手", "finger": "1 拇指", "press_s": 2.75, "release_s": 2.83, "after_cut": true}], "measured": {"dx_white": -0.01, "horizon": 0.5}, "frames": ["t = 0.53 秒", "t = 0.67 秒", "t = 0.80 秒", "t = 0.93 秒", "t = 1.07 秒", "t = 1.20 秒", "t = 1.27 秒", "t = 1.40 秒", "t = 1.53 秒", "t = 1.67 秒", "t = 1.80 秒", "t = 1.93 秒"]}

## EXAMPLE 6 OUTPUT

{"stem": "请推测：接下来的 0.5 秒里，右手整只手会朝哪个方向移动？", "explanation": "切点前右手约 1.1 秒用中指按 C#5、约 1.9 秒用无名指按 D5，手一直停在同一位置；切点之后要到约 2.8 秒才会用拇指按 A4，0.5 秒内右手的位置几乎不变。所以答案是基本不动。", "visual_notes": "开头到结尾，右手一直停在中央 C 右侧的同一位置，手掌没有向两侧平移，只有手指交替按键。"}

