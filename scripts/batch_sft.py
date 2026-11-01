"""Resumable PianoVAM SFT production, reusing the upstream exporters and generators.

Private labels never enter the visual observation pass. Final-test recordings are
inventoried but never sent to the API. All outputs live under --out.
"""
import argparse
import base64
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MEDIA_LOCK = threading.Lock()
PROFILE = "vam-core-sft-v2"
BASELINE = "37169e51415b2792e3497a41c354fc778d2317c1"
CORE = [("T1", s) for s in ("按键序列", "音区", "旋律走向", "织体", "和弦与音程")]
CORE += [("T2", "哪只手"), ("T3", "按键次数（手）"),
         ("T4", "旋律定位"), ("T4", "和弦定位"), ("T5", "会不会按键")]
CONVENTIONS = """左手和右手均指演奏者本人的左右手。画面左侧是低音方向。
C4是中央C，黑键用升号。时间均为当前交付片段相对秒数；区间左闭右开。
一次按键是一个键的一次按下；同时按多个键分别计数，重复按同一键也分别计数。
同时按下的分组以本组最早按下时刻为起点，40毫秒内的按下归同一组，不链式延长。
不要推断手指身份、穿指跨指、悬空、手指持续接触或踏板状态。"""
OBSERVE = """【任务】只根据题干与连续交付画面检查画面是否可供复核，不回答题目。
【约定】""" + CONVENTIONS + """
【检查字段】
1. visible：所问时段的画面是否清楚、连续，布尔值true或false。
2. keyboard_visible：目标琴键所在的键盘区域是否可见，布尔值true或false。
3. target_region_visible：题目指定的手部区域是否可辨；问哪只手时须能区分两只手，布尔值true或false。
【输出】只输出上述三个字段的JSON，不写任何动作、时刻、按键、接触状态或未来预测的自由描述。
【示例】{"visible":true,"keyboard_visible":true,"target_region_visible":true}"""
WRITE = """【任务】从程序证据中选择与本题有关的证据ID，并选择组织顺序。
【约定】""" + CONVENTIONS + """
【输入】question是固定题干；answer是真值；atoms为带atom_id的程序事实。不得添加不存在的事实。
【输出】只输出JSON：{"atom_ids":["e00001","e00002"],"style":"chronological"}。
1. atom_ids最多12个，必须来自输入。存在key_press原子时至少选择一个；尽量选择直接支持问题的全部相关按下事件。
2. style只能是chronological（按时序）或concise（简洁叙述）。
3. 不输出分析自由文本、画面动作描述、手指身份或最终答案。程序负责把引用的真实原子写成完整证据句和固定答案。
4. 预测题的原子只包含已见历史；不能选取或虚构未来证据。
【示例】{"atom_ids":["e00001"],"style":"concise"}"""
JUDGE = """【任务】核对钢琴问答训练样本是否忠实于给定事实与观察。
【约定】""" + CONVENTIONS + """
【输出】仅输出JSON：{"hard_error":false,"faithful":9,"clear":9,"comment":"简短理由"}。
1. hard_error必须是布尔值，faithful和clear必须是0到10的整数。
2. 手、键、计数、时间、区间、选项与答案关系错误，或者添加未经证实的手指/接触状态，hard_error=true，一票否决。
3. 预测题将未来事实冒充已看到的线索，hard_error=true。已知真值只用于核对结尾，不能当观察证据。
4. 证据不足、解析与问题无关或画面观察与事实明显冲突时，不应打高分。不要为了标准答案编理由。
5. 你只做文字一致性核对，不宣称已经独立检查原视频；画面可见性由前一轮及后续人工核验负责。"""
FORBIDDEN = ("拇指", "食指", "中指", "无名指", "小指", "悬空", "穿指", "跨指", "踏板", "记录", "数据", "事实卡", "标注")
KEY_RE = re.compile(r"(?<![A-Za-z])[A-G]#?[0-8](?!\d)")
SEC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*秒")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    default=str).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def answer_text(q):
    a = q["answer"]
    return ", ".join(str(x) for x in a) if isinstance(a, list) else str(a)


def final_answer(observations, analysis, q):
    obs = "".join(observations)
    ana = "".join(x["text"] for x in analysis)
    if not obs or not ana or any(x in obs + ana for x in ("【观察】", "【分析】", "所以答案是")):
        raise ValueError("invalid SFT body")
    return "【观察】" + obs + "【分析】" + ana + "所以答案是{" + answer_text(q) + "}"


def validate_analysis(value, atoms, q):
    steps = value.get("analysis")
    if not isinstance(steps, list) or not 1 <= len(steps) <= 4:
        raise ValueError("analysis must contain 1-4 cited clauses")
    by_id = {a["atom_id"]: a for a in atoms}
    for step in steps:
        ids, text = step.get("atom_ids"), step.get("text")
        if (not isinstance(ids, list) or not ids or any(i not in by_id for i in ids)
                or not isinstance(text, str) or not 5 <= len(text) <= 160):
            raise ValueError("invalid evidence citation")
        blocked = [w for w in FORBIDDEN if w in text]
        if blocked or "所以答案是" in text:
            raise ValueError("remove unsupported/source terms: " + ",".join(blocked) + "; final answer is written by code")
        refs = [by_id[i] for i in ids]
        allowed_keys = {a["key"] for a in refs if "key" in a}
        if set(KEY_RE.findall(text)) - allowed_keys:
            raise ValueError("key not supported by cited atoms")
        for hand in ("左手", "右手"):
            if hand in text and not any(a.get("hand") == hand or a.get("kind") == "visual_context" for a in refs):
                raise ValueError("hand not supported by cited atoms")
        for sec in SEC_RE.findall(text):
            if len(sec.partition(".")[2]) > 1:
                raise ValueError("time precision must be one decimal")
            if not any(abs(float(sec) - a[k]) <= 0.08 for a in refs for k in ("press_s", "release_s") if k in a):
                raise ValueError("time not supported by cited atoms")
        if q["task"] == "T5" and re.search(r"切点后|切点之后|之后.{0,12}(没有|未|已经|发生|按下)", text):
            raise ValueError("future event used as prediction evidence")
    return steps


def program_summary(q, atoms):
    """A typed conclusion derived from the generator's truth, never from model prose."""
    content = (q["options"][ord(q["answer"]) - 65] if q["answer_type"] == "choice" else q["answer"])
    task, subtype = q["task"], q["subtype"]
    if task == "T3":
        count = sum(a.get("kind") == "key_press" for a in atoms)
        if int(content) != count:
            raise ValueError("counter answer does not equal the number of event atoms")
        text = f"所问区间内共有{count}次按下事件，同时按下的不同键分别计数。"
    elif task == "T2":
        target = q.get("extra", {}).get("key")
        if target is None:
            target = KEY_RE.findall(q["question"])[0]
        events = [a for a in atoms if a.get("key") == target]
        if len(events) != 1 or events[0]["hand"] != content:
            raise ValueError("which-hand truth does not match the target event atom")
        text = f"目标键{target}由{content}按下。"
    elif task == "T4":
        if isinstance(content, list):
            text = f"该段旋律从首音按下到末音松开，对应约{content[0]:.1f}秒至{content[1]:.1f}秒。"
        else:
            text = f"所问和弦的首次按下时刻约为{float(content):.1f}秒。"
    elif task == "T5":
        text = f"根据截至{q['predict']['cut']:.1f}秒的已见片段，对接下来{q['predict']['horizon']:.1f}秒是否出现新一次按键作短时预测。"
    elif subtype == "按键序列":
        seq = " → ".join(content) if isinstance(content, list) else str(content)
        text = f"按下的键依次为{seq}。"
    elif subtype == "音区":
        zone = str(content) if q["answer_type"] == "choice" else f"第{content}组（C{content}–B{content}）"
        text = f"区间中按下的键主要位于{zone}。"
    elif subtype == "旋律走向":
        text = f"所问音符序列的整体走向为{content}。"
    elif subtype == "织体":
        text = f"按照题目采用的同时按下分组规则，区间内的织体为：{content}。"
    else:
        text = f"相应同时按下的键构成{content}。"
    return dict(atom_id="program_summary", kind="task_summary", text=text)


def controlled_analysis(value, atoms, q):
    if set(value) != {"atom_ids", "style"} or value["style"] not in ("chronological", "concise"):
        raise ValueError("output exactly atom_ids and style; do not write free text")
    ids = value["atom_ids"]
    by_id = {a["atom_id"]: a for a in atoms}
    if (not isinstance(ids, list) or len(ids) > 12 or any(not isinstance(i, str) or i not in by_id for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("invalid or invented evidence IDs")
    events = [a for a in atoms if a.get("kind") == "key_press"]
    chosen = [by_id[i] for i in ids if by_id[i].get("kind") == "key_press"]
    if events and not chosen:
        raise ValueError("select at least one real key_press evidence ID")
    if q["task"] == "T3":
        chosen = events  # A count explanation cannot silently omit counted events.
    steps = []
    for atom in sorted(chosen, key=lambda a: a["press_s"]):
        if value["style"] == "chronological":
            text = f"{atom['hand']}在约{atom['press_s']:.1f}秒按下{atom['key']}。"
        else:
            text = f"约{atom['press_s']:.1f}秒时，{atom['hand']}按下{atom['key']}。"
        steps.append(dict(atom_ids=[atom["atom_id"]], text=text))
    summary = by_id.get("program_summary")
    if summary is None:
        raise ValueError("missing program-derived task summary")
    steps.append(dict(atom_ids=[summary["atom_id"]], text=summary["text"]))
    return steps


def judge_pass(j):
    return (type(j.get("hard_error")) is bool and not j["hard_error"]
            and all(type(j.get(k)) is int and 0 <= j[k] <= 10 for k in ("faithful", "clear"))
            and min(j["faithful"], j["clear"]) >= 8)


def shuffle_choices(q, rng):
    """Shuffle the production copy, including fixed-order upstream options."""
    if q["answer_type"] != "choice":
        return q
    old = list(q["options"])
    order = [int(x) for x in rng.permutation(len(old))]
    original = ord(q["answer"]) - 65
    lines = q["question"].split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("A. "))
    q["options"] = [old[i] for i in order]
    q["answer"] = chr(65 + order.index(original))
    lines[start:start + len(old)] = [f"{chr(65 + i)}. {v}" for i, v in enumerate(q["options"])]
    q["question"] = "\n".join(lines)
    copy = q.get("baselines", {}).get("copy")
    if isinstance(copy, str) and len(copy) == 1 and 0 <= ord(copy) - 65 < len(old):
        q["baselines"]["copy"] = chr(65 + order.index(ord(copy) - 65))
    return q


def grid_offsets(n, fps, output_fps, prediction=False):
    out, k = [], 0
    while True:
        off = round(k * fps / output_fps)
        if off >= n:
            break
        if not out or out[-1] != off:
            out.append(off)
        k += 1
    if prediction and (not out or out[-1] != n - 1):
        out.append(n - 1)
    return out


def partition(entries, gold, dev_count):
    groups = {}
    for e in entries:
        # A composition name alone does not identify a duplicate performance.
        duplicate = e.get("duplicate_group")
        groups[e["record_time"]] = (("duplicate", str(duplicate)) if duplicate
                                    else ("record", e["record_time"]))
    held_groups = {groups[v] for v in gold if v in groups}
    remaining = [v for v in groups if v not in gold and groups[v] not in held_groups]
    group_order = sorted({groups[v] for v in remaining}, key=digest)
    dev_groups, count = set(), 0
    for g in group_order:
        if count >= dev_count:
            break
        dev_groups.add(g)
        count += sum(groups[v] == g for v in remaining)
    return {v: ("final_test" if v in gold else "heldout_related" if groups[v] in held_groups
                else "dev" if groups[v] in dev_groups else "train") for v in groups}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


class ModelReplyError(ValueError):
    def __init__(self, message, trace):
        super().__init__(message)
        self.trace = trace


def decode_reply(result, model, started, attempt):
    choice = result["choices"][0]
    raw = choice["message"].get("content") or ""
    trace = dict(model=model, raw=raw, usage=result.get("usage"),
                 seconds=round(time.monotonic() - started, 3), retries=attempt)
    if choice["finish_reason"] != "stop":
        raise ModelReplyError("truncated model reply", trace)
    match = re.fullmatch(r"\s*(?:```(?:json)?\s*)?(\{.*\})\s*(?:```)?\s*", raw, re.S)
    if not match:
        raise ModelReplyError("non-JSON model reply", trace)
    try:
        value = json.loads(match[1])
    except ValueError:
        raise ModelReplyError("invalid JSON model reply", trace) from None
    return value, trace


class API:
    HOSTS = {"Qwen3.8-27B": "cqhbod8bjjjbcoakk8pmeebgkaq9akcq",
             "dsv4-pro-0813": "gb588hkmdmpgcagomjcqmbeb9cmakjmq"}

    def __init__(self, doc):
        text = Path(doc).read_text(encoding="utf-8")
        self.auth = {}
        for model, host in self.HOSTS.items():
            pos = text.index(host + ".openapi-sj.sii.edu.cn/v1/chat/completions")
            keys = list(re.finditer(r'(?:INF_API_KEY|API_KEY)\s*=\s*["\x27]([^"\x27\r\n]+)', text[:pos], re.I))
            if not keys:
                raise ValueError("no matching institute API configuration")
            self.auth[model] = ("https://" + host + ".openapi-sj.sii.edu.cn/v1/chat/completions", keys[-1].group(1))

    def call(self, model, system, content, max_tokens=1400):
        url, key = self.auth[model]
        body = dict(model=model, stream=False, max_tokens=max_tokens, temperature=0,
                    chat_template_kwargs={"thinking": False} if model.startswith("ds") else {"enable_thinking": False},
                    messages=[dict(role="system", content=system), dict(role="user", content=content)])
        opener = urllib.request.build_opener(NoRedirect())
        for attempt in range(3):
            t = time.monotonic()
            try:
                req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
                with opener.open(req, timeout=180) as response:
                    result = json.load(response)
                return decode_reply(result, model, t, attempt)
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == 2:
                    try:
                        error = json.loads(e.read(4096)).get("error", {})
                        message = str(error.get("message", "")) if isinstance(error, dict) else ""
                        message = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", message.replace(key, "[REDACTED]"))
                    except Exception:
                        message = ""
                    raise RuntimeError("institute API HTTP " + str(e.code) + ": " + message[:160]) from None
            except (TimeoutError, urllib.error.URLError):
                if attempt == 2:
                    raise RuntimeError("institute API timed out/unreachable") from None
            time.sleep(min(10, 2 ** (attempt + 1)))


def inventory(a, root):
    raw = read_json(Path(a.raw) / "metadata.json")
    entries = list(raw.values()) if isinstance(raw, dict) else raw
    ids = {e["record_time"] for e in entries}
    if len(ids) != len(entries):
        raise ValueError("duplicate recording IDs in metadata")
    unindexed = sorted(p.stem for p in (Path(a.raw) / "Video").glob("*.mp4") if p.stem not in ids)
    gold = set(Path(a.gold_ids).read_text().split())
    if len(gold) != 11:
        raise ValueError("expected frozen 11-video final-test list")
    splits = partition(entries, gold, a.dev_videos)
    records = []
    for e in sorted(entries, key=lambda x: x["record_time"]):
        v = e["record_time"]
        paths = [Path(a.raw) / "Video" / (v + ".mp4"), Path(a.raw) / "Handskeleton" / (v + ".json"),
                 Path(a.labels) / "Fingering" / (v + ".tsv")]
        missing = [str(p) for p in paths if not p.is_file()]
        records.append(dict(video_id=v, split=splits[v], missing=missing, metadata=e,
                            inputs=[dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                    for p in paths if p.exists()]))
    code_files = [Path(__file__), *(ROOT / "pmfinger").glob("*.py"),
                  ROOT / "scripts/export_pianovam.py", ROOT / "scripts/run_fingering.py",
                  *(ROOT / "models").glob("*.npz")]
    code_hash = digest({str(p.relative_to(ROOT)): file_hash(p) for p in code_files})
    plan = dict(profile=PROFILE, upstream_commit=BASELINE, code_sha256=code_hash,
                per_subtype=a.per_subtype, max_tries=a.max_tries, seed=a.seed, core=CORE, writer="Qwen3.8-27B", judge="dsv4-pro-0813",
                prompt_sha256=digest([OBSERVE, WRITE, JUDGE]), gold_ids=sorted(gold),
                unindexed_videos=unindexed, records=records)
    old = root / "manifest.json"
    if old.exists() and digest(read_json(old)) != digest(plan):
        raise ValueError("frozen run manifest changed; select a new output directory")
    atomic_json(old, plan)
    data = root / "data"
    data.mkdir(exist_ok=True)
    for name, target in [("Video", Path(a.raw) / "Video"), ("Handskeleton", Path(a.raw) / "Handskeleton"),
                         ("Fingering", Path(a.labels) / "Fingering"),
                         ("Fingering_GT", Path(a.labels) / "Fingering_GT")]:
        link = data / name
        if link.is_symlink() and link.resolve() != target.resolve():
            raise ValueError("unexpected data symlink")
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
    return plan


def prepare_video(a, root, record):
    import numpy as np
    from export_pianovam import export
    from run_fingering import work
    from pmfinger.states import load_derived
    from check_octave import hit_rate, SHIFTS
    vid = record["video_id"]
    perf_dir, fing_dir = root / "perf", root / "fing"
    perf_dir.mkdir(exist_ok=True)
    fing_dir.mkdir(exist_ok=True)
    pp, fp = perf_dir / (vid + ".npz"), fing_dir / (vid + ".fing.npz")
    receipt = root / "receipts" / (vid + ".json")
    if not receipt.exists():
        export(str(root / "data"), vid, record["metadata"], str(perf_dir))
        # Recompute with the frozen upstream model/config; no unverified old HMM cache reuse.
        overrides = ["kernel=tri", "sigma_white=0.56", "sigma_black=0.56", "w_seq=0.5",
                     "conf_thresh=0.7", "release_lateral=0.35", "av_search=12"]
        res = work((str(pp), str(ROOT / "models/prior.npz"), str(fing_dir), overrides, str(ROOT / "models/calib.npz")))
        atomic_json(receipt, dict(result=res, perf_sha256=file_hash(pp), fing_sha256=file_hash(fp)))
    saved = read_json(receipt)
    if file_hash(pp) != saved["perf_sha256"] or file_hash(fp) != saved["fing_sha256"]:
        raise ValueError("derived cache hash mismatch")
    d = load_derived(str(pp), str(fp))
    with np.load(pp) as z, np.load(fp) as r:
        if (not np.array_equal(z["note_pitch"], r["pitch"])
                or not np.allclose(z["note_onset"], r["onset"], atol=1e-9, rtol=0)
                or int(z["frame_offset"]) != 0
                or not np.array_equal(d.on, np.rint(z["note_onset"] * d.fps).astype(int) + d.av)):
            raise ValueError("performance/result timeline mismatch")
        if "note_key_offset" not in z or not np.isfinite(z["note_key_offset"]).all() or (z["note_key_offset"] < z["note_onset"]).any():
            raise ValueError("missing real key release")
        from p0_restore import state
        d.hand, d.hand_src, _ = state.trust_sources(dict(z), dict(r))
        d.hand_trust = d.hand >= 0
        # A rounded zero-frame press is invisible, not an invented one-frame hold.
        d.rel = np.rint(z["note_key_offset"] * d.fps).astype(int) + d.av
        # No finger subtype is enabled; also suppress automatic finger claims in cards.
        d.finger_trust[:] = False
    qc = root / "qc" / (vid + ".json")
    if not qc.exists():
        rates = {}
        for shift in SHIFTS:
            ps = d.pitch + shift
            keep = (ps >= 21) & (ps <= 108)
            rates[str(shift)] = max(hit_rate(d.tips, d.ok, ps[keep], d.onset_s[keep], d.fps, av)
                                    for av in range(-12, 13))
        passed = rates["0"] > max(v for k, v in rates.items() if k != "0") and rates["0"] - max(rates[str(x)] for x in (-24, -12, 12, 24)) >= 0.15
        atomic_json(qc, dict(ok=bool(passed), rates=rates, av=d.av))
    if not read_json(qc)["ok"]:
        raise ValueError("whole-video keyboard/octave QC failed")
    return d


def generate(a, root, record, d):
    import numpy as np
    from pmfinger.bench import GENERATORS, BenchConfig, Sampler, expand
    from gen_bench import overlaps
    from pmfinger.states import chord_name
    vid = record["video_id"]
    path = root / "facts" / (vid + ".json")
    if path.exists():
        return read_json(path)
    seed = int(digest([a.seed, vid])[:8], 16)
    rng = np.random.default_rng(seed)
    cfg = BenchConfig(range_types=("timestamp",), group_prob=0, describe_prob=0)
    rows, counts = [], {}
    for task, sub in CORE:
        got = []
        for attempt in range(a.max_tries):
            if len(got) >= a.per_subtype:
                break
            f = GENERATORS[task, sub](Sampler(d, rng, cfg))
            if f is None or any(overlaps(f, old) for old in got):
                continue
            if f["task"] == "T1" and f["subtype"] == "和弦":
                # Symmetric chords have equivalent roots: keep only unambiguous names.
                if any(s in f["mc"]["options"][ord(f["mc"]["answer"]) - 65] for s in ("增三", "减七")):
                    continue
            if task == "T4":
                h, sp = f["hand"], f["span"]
                groups = d.events(d.notes_of(h, sp["c0"], sp["c1"]))
                if sub == "和弦定位":
                    group = next(e for e in groups if len(e) >= 3)
                    f["fill"]["answer"] = round((min(d.on[group]) - sp["c0"]) / d.fps, 3)
                else:
                    lo, hi = f["fill"]["answer"]
                    if not 0.3 < lo < hi < sp["L"] - 0.3:
                        continue
            if task == "T5":
                f["body"] = f["body"].replace("按下新的键", "发生新一次按键（重复按同一键也算）")
            # One canonical question per fact: never reuse analysis across time bases.
            questions = expand(d, f, rng, cfg)
            if not questions:
                continue
            q = shuffle_choices(questions[0], rng)
            q["qid"] = PROFILE + "-" + f["fact_id"]
            q["semantic_id"] = PROFILE + "/" + task + "/" + sub
            q["split"] = record["split"]
            if task == "T5":
                q["clip"]["end_frame"] += 1  # include the actual now frame
                q["clip"]["length_s"] = (q["clip"]["end_frame"] - q["clip"]["start_frame"]) / d.fps
            got.append(f)
            rows.append(dict(fact=f, question=q))
        counts[task + "/" + sub] = len(got)
    atomic_json(path, rows)
    atomic_json(root / "counts" / (vid + ".json"), counts)
    return rows


def frame_pack(root, q, d):
    import cv2
    import numpy as np
    c0, c1, fps = q["clip"]["start_frame"], q["clip"]["end_frame"], d.fps
    pred = q["task"] == "T5"
    offs = ([int(t) - c0 for t in q["delivery_frames"]] if "delivery_frames" in q
            else grid_offsets(c1 - c0, fps, q["frame_fps"], pred))
    if not offs or offs != sorted(set(offs)) or offs[0] != 0 or offs[-1] >= c1-c0:
        raise ValueError("invalid explicit delivery grid")
    if pred and abs(offs[-1] / fps - q["predict"]["cut"]) > 1e-8:
        raise ValueError("last frame is not the stated prediction now")
    folder = root / "media" / q["clip_id"]
    manifest = folder / "manifest.json"
    if manifest.exists():
        m = read_json(manifest)
        if m["offsets"] != offs or any(not (root / p["path"]).is_file() or file_hash(root / p["path"]) != p["sha256"] for p in m["packs"] + m["frames"]):
            raise ValueError("frame cache integrity mismatch")
        return m
    folder.mkdir(parents=True, exist_ok=True)
    meta = read_json(root / "perf" / (d.vid + ".meta.json"))
    W, H = meta["width"], meta["height"]
    corners = np.array(meta["keyboard_corners_px"])
    kh = float(corners[:, 1].max() - corners[:, 1].min())
    # The crop uses only frames actually delivered, and retains both hands and all keys.
    with np.load(root / "perf" / (d.vid + ".skeleton.npz")) as z:
        sk = z["skeleton"][c0:c1]
        yy = sk[..., 1][np.isfinite(sk[..., 1])]
    y0 = max(0, int(min(corners[:, 1].min() - kh * 0.4, yy.min() - 20 if len(yy) else H)))
    y1 = min(H, int(max(corners[:, 1].max() + kh * 1.1, yy.max() + 20 if len(yy) else 0)))
    cap = cv2.VideoCapture(d.perf.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, c0)
    if abs(cap.get(cv2.CAP_PROP_POS_FRAMES) - c0) > 0.1:
        cap.release()
        raise ValueError("video seek mismatch")
    tiles, frames, packs = [], [], []
    wanted = set(offs)
    try:
        for off in range(offs[-1] + 1):
            ok, image = cap.read()
            if not ok:
                raise ValueError("video ended before requested frame")
            if off not in wanted:
                continue
            im = image[y0:y1]
            scale = min(1.0, 1280 / max(im.shape[:2]))
            if scale < 1:
                im = cv2.resize(im, (round(im.shape[1] * scale), round(im.shape[0] * scale)))
            tile = np.full((im.shape[0] + 26, im.shape[1], 3), 255, dtype=np.uint8)
            tile[26:] = im
            cv2.putText(tile, f"t={off / fps:.3f}s | native={c0 + off}", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            tile_path = folder / f"frame_{off:05d}.jpg"
            if not cv2.imwrite(str(tile_path), tile, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise ValueError("failed to save delivery frame")
            frames.append(dict(path=str(tile_path.relative_to(root)), t=off / fps,
                               native_frame=c0 + off, sha256=file_hash(tile_path)))
            tiles.append(tile)
        for i in range(0, len(tiles), 3):
            im = np.concatenate(tiles[i:i + 3], axis=0)
            p = folder / f"pack_{i // 3:03d}.jpg"
            if not cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise ValueError("failed to save image pack")
            packs.append(dict(path=str(p.relative_to(root)), sha256=file_hash(p)))
    finally:
        cap.release()
    m = dict(offsets=offs, frames=frames, packs=packs, crop=[0, y0, W, y1],
             clip_start_native=c0, clip_end_native_exclusive=c1)
    atomic_json(manifest, m)
    return m


def check_event_visibility(on, rel, pitch, onset_s, groups, delivered, fps, group_sensitive):
    """Reject unresolvable repetitions and 40 ms chord/sequence boundaries."""
    import numpy as np
    ff = np.asarray(delivered)
    for key in set(int(pitch[n]) for g in groups for n in g):
        ns = sorted((n for g in groups for n in g if int(pitch[n]) == key), key=lambda n: on[n])
        for n, nxt in zip(ns, ns[1:]):
            if not ((ff >= rel[n]) & (ff < on[nxt])).any():
                raise ValueError("repeated key lacks a delivered released frame between presses")
    if not group_sensitive:
        return
    for i, group in enumerate(groups):
        first = min(group, key=lambda n: onset_s[n])
        comparisons = [n for n in group if n != first]
        if i + 1 < len(groups):
            comparisons.append(min(groups[i + 1], key=lambda n: onset_s[n]))
        if any(abs((onset_s[n] - onset_s[first]) - 0.04) <= 1 / fps for n in comparisons):
            raise ValueError("onset grouping lies within one native frame of the 40 ms boundary")
        if i + 1 < len(groups):
            last_on = max(on[n] for n in group)
            next_on = min(on[n] for n in groups[i + 1])
            if not ((ff >= last_on) & (ff < next_on)).any():
                raise ValueError("successive onset groups cannot be distinguished on delivered frames")


def evidence(q, d, frames):
    import numpy as np
    h = q["tags"].get("hand")
    c0 = q["clip"]["start_frame"]
    if q["range"]:
        a = c0 + round(q["range"]["start"] * d.fps)
        b = c0 + round(q["range"]["end"] * d.fps)
    else:
        a, b = c0, q["clip"]["end_frame"]
    if q["task"] == "T5":
        b = c0 + round(q["predict"]["cut"] * d.fps) + 1
    idx = np.flatnonzero((d.on >= a) & (d.on < b) & d.hand_trust)
    if h is not None:
        idx = idx[d.hand[idx] == h]
    from pmfinger.states import HAND_ZH, pitch_name
    atoms = []
    ff = np.array([f["native_frame"] for f in frames])
    # P0 gates apply to every enabled key subtype, including historical evidence for prediction.
    from p0_restore import state, retrospective_gate
    hand = np.where(d.hand_trust, d.hand, -1)
    raw = state.Perf(d.vid, d.fps, d.T, d.perf.video_path, d.pitch, d.on, d.rel,
                     hand, np.full(d.N, "v2_auto_agreement"), np.full(d.N, -1),
                     np.full((d.T, 2), np.nan), np.full((d.T, 2, 5), np.nan), d.ok, d.onset_s)
    sensitive = q["task"] == "T4" or (q["task"] == "T1" and q["subtype"] != "音区")
    for target_hand in ([h] if h is not None else [0, 1]):
        retrospective_gate(raw, target_hand, a, b, ff, sensitive)
    for n in idx:
        # For retrospective key questions at least one delivered frame must show the press held.
        if q["task"] != "T5" and not ((ff >= d.on[n]) & (ff < d.rel[n])).any():
            raise ValueError("a target key press falls wholly between delivered frames")
        atom = dict(atom_id=f"e{int(n):05d}", kind="key_press", hand=HAND_ZH[int(d.hand[n])],
                    key=pitch_name(d.pitch[n]), press_s=float((d.on[n] - c0) / d.fps))
        release = float((d.rel[n] - c0) / d.fps)
        if q["task"] != "T5" or release <= q["predict"]["cut"]:
            atom["release_s"] = release
        atoms.append(atom)
    atoms.append(dict(atom_id="visual_context", kind="visual_context", hand=HAND_ZH[h] if h is not None else None,
                      text="连续交付帧中的键盘和演奏者双手；仅用于引用画面观察，不支持精确事件时刻。"))
    return atoms


def produce_one(api, root, row, d, plan):
    q = row["question"]
    result_path = root / "results" / (q["qid"] + ".json")
    record = dict(qid=q["qid"], video_id=d.vid, task=q["task"], subtype=q["subtype"], split=q["split"],
                  question=q, status="needs_review", api=[])
    try:
        with MEDIA_LOCK:
            media = frame_pack(root, q, d)
        atoms = evidence(q, d, media["frames"])
        atoms.append(program_summary(q, atoms))
        key = digest([q, media, atoms, plan["code_sha256"], plan["prompt_sha256"]])
        if result_path.exists():
            old = read_json(result_path)
            if old.get("input_sha256") != key:
                raise ValueError("completed input changed")
            return old
        record.update(input_sha256=key, atoms=atoms, media=media)
        content = [dict(type="text", text=q["question"])]
        for item in media["packs"]:
            uri = "data:image/jpeg;base64," + base64.b64encode((root / item["path"]).read_bytes()).decode()
            content.append(dict(type="image_url", image_url=dict(url=uri)))
        stage = "B"
        obs, raw = api.call("Qwen3.8-27B", OBSERVE, content, 700)
        record["api"].append(dict(stage="B", **raw))
        fields = ("visible", "keyboard_visible", "target_region_visible")
        if set(obs) != set(fields) or any(type(obs.get(k)) is not bool or not obs[k] for k in fields):
            raise ValueError("visual pass uncertain or malformed structured fields")
        observations = ["交付画面中可见键盘，所问手部所在区域清晰可辨。"]
        card = dict(question=q["question"], task=q["task"], answer=answer_text(q),
                    answer_content=q["options"][ord(q["answer"]) - 65] if q["answer_type"] == "choice" else q["answer"],
                    atoms=atoms, observations=observations)
        for attempt in range(2):
            stage = "D"
            out, raw = api.call("Qwen3.8-27B", WRITE, json.dumps(card, ensure_ascii=False))
            record["api"].append(dict(stage="D", **raw))
            try:
                analysis = controlled_analysis(out, atoms, q)
                break
            except ValueError as e:
                if attempt:
                    raise
                card["previous_error"] = str(e)
        answer = final_answer(observations, analysis, q)
        stage = "judge"
        judged, raw = api.call("dsv4-pro-0813", JUDGE, json.dumps(dict(**card, analysis=analysis, sft_answer=answer), ensure_ascii=False), 700)
        record["api"].append(dict(stage="judge", **raw))
        record.update(observation=obs, analysis=analysis, judge=judged)
        if not judge_pass(judged):
            raise ValueError("judge hard error / low or malformed scores")
        record.update(status="auto_accepted", answer=answer, human_verified=False)
    except Exception as e:
        # Preserve usage/raw replies even when JSON decoding failed.
        if isinstance(e, ModelReplyError):
            record["api"].append(dict(stage=stage, **e.trace))
        record["error"] = type(e).__name__ + ": " + str(e)[:200]
    atomic_json(result_path, record)
    return record


def summarize(root, state):
    counts, tasks, usage = collections.Counter(), collections.Counter(), collections.Counter()
    exports = {s: [] for s in ("train", "dev", "needs_review")}
    for p in sorted((root / "results").glob("*.json")):
        r = read_json(p)
        counts[r["status"]] += 1
        tasks[r["task"] + "/" + r["status"]] += 1
        for call in r.get("api", []):
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[k] += (call.get("usage") or {}).get(k, 0)
        if r["status"] == "auto_accepted":
            messages = [dict(role="system", content=CONVENTIONS + "\n请按【观察】…【分析】…所以答案是{X}作答。"),
                        dict(role="user", content=r["question"]["question"]), dict(role="assistant", content=r["answer"])]
            exports[r["split"]].append(dict(id=r["qid"], video_id=r["video_id"], messages=messages,
                                             images=[m["path"] for m in r["media"]["packs"]],
                                             task=r["task"], subtype=r["subtype"], human_verified=False,
                                             input_sha256=r["input_sha256"]))
        else:
            exports["needs_review"].append(dict(id=r["qid"], video_id=r["video_id"], error=r.get("error"),
                                               result=str(p.relative_to(root))))
    for split, rows in exports.items():
        dest = root / ("sft_" + split + ".jsonl")
        tmp = dest.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        os.replace(tmp, dest)
    status = dict(state=state, updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  results=dict(counts), tasks=dict(tasks), usage=dict(usage),
                  train=len(exports["train"]), dev=len(exports["dev"]), needs_review=len(exports["needs_review"]))
    atomic_json(root / "status.json", status)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "labels", "gold-ids", "out", "api-doc"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--per-subtype", type=int, default=4)
    ap.add_argument("--max-tries", type=int, default=600)
    ap.add_argument("--dev-videos", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = Path(a.out).resolve()
    protected = Path("/inspire/hdd/global_user/xuguangjun-253208580299")
    if str(root).startswith(str(protected)) and not root.is_relative_to(protected / "users/qihengwang"):
        raise ValueError("production output must stay in the authorized own directory")
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = open(root / "batch.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = inventory(a, root)
    api = API(a.api_doc)
    records = sorted((r for r in plan["records"] if r["split"] in ("train", "dev") and not r["missing"]),
                     key=lambda r: (r["split"] != "dev", r["video_id"]))
    completed = 0
    failures = []
    summarize(root, "running")
    for rec in records:
        print("VIDEO_START", rec["video_id"], rec["split"], flush=True)
        try:
            d = prepare_video(a, root, rec)
            rows = generate(a, root, rec, d)
            if a.limit:
                rows = rows[:max(0, a.limit - completed)]
            with ThreadPoolExecutor(a.workers) as ex:
                for fut in as_completed([ex.submit(produce_one, api, root, row, d, plan) for row in rows]):
                    r = fut.result()
                    completed += 1
                    if any(x in r.get("error", "") for x in ("HTTP 401", "HTTP 403")):
                        raise RuntimeError("API authorization failed; stop bulk calls")
                    print("RESULT", r["qid"], r["status"], r.get("error", ""), flush=True)
                    if completed <= 4 or completed % 10 == 0:
                        summarize(root, "running")
            summarize(root, "running")
            print("VIDEO_DONE", rec["video_id"], len(rows), flush=True)
        except Exception as e:
            if "API authorization failed" in str(e):
                summarize(root, "api_authorization_failed")
                raise SystemExit(str(e))
            failures.append(dict(video_id=rec["video_id"], error=type(e).__name__ + ": " + str(e)[:200]))
            atomic_json(root / "video_errors.json", failures)
            print("VIDEO_ERROR", rec["video_id"], type(e).__name__, str(e)[:200], flush=True)
            if a.limit and len(failures) >= 2:
                summarize(root, "smoke_failed")
                raise SystemExit("smoke preprocessing failed on two recordings")
        if a.limit and completed >= a.limit:
            break
    summarize(root, "smoke_finished" if a.limit else "finished_with_video_errors" if failures else "finished")


if __name__ == "__main__":
    main()
