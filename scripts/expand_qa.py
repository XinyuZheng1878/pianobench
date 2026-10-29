"""Expansion steps B-C: let an LLM rewrite each fact card into a natural question stem + explanation,
check every output automatically, then rebuild the question set with the new wording.

  # B. generate (any OpenAI-compatible endpoint; frames are sent for subtypes that need them)
  python -B scripts/expand_qa.py run --cards work/cards.jsonl --release work/release \
      --base-url https://api.openai.com/v1 --model gpt-5-mini --out work/expansions.jsonl [--limit 20]
  python -B scripts/expand_qa.py run ... --mock          # offline test, no API
  python -B scripts/expand_qa.py run ... --show 3        # print the first 3 requests, send nothing

  # B2. quality scoring by a second LLM pass (SVBench-style); low scores go to human review
  python -B scripts/expand_qa.py judge --cards work/cards.jsonl --expansions work/expansions.jsonl \
      --out work/judgments.jsonl --base-url ... --model ...

  # C. apply: questions.jsonl with LLM wording (answers / options / instructions untouched)
  python -B scripts/expand_qa.py apply --release work/release --cards work/cards.jsonl \
      --expansions work/expansions.jsonl --out work/release/questions_nl.jsonl

Division of labour: the LLM writes only `stem` (the question body, with 【区间】 where the interval goes),
`explanation` and `visual_notes`. Answers, options and answer instructions are copied from the checked
facts by code, so the LLM can make a question read better but can never change what is right.
Every output goes through validate(); a failing output is sent back with the reasons (up to --retries
times), and a fact that never passes keeps its template wording and is listed in the summary.
"""
import argparse
import base64
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

PH = "【区间】"
KEY_RE = re.compile(r"(?<![A-Za-z])[A-G]#?[0-8](?!\d)")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
SEC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*秒")
HANDS = ["左手", "右手"]
FINGERS = ["拇指", "食指", "中指", "无名指", "小指"]
SOURCE_WORDS = ["事实卡", "记录", "数据", "标注", "MIDI", "JSON", "events", "measured", "第几张", "图片", "张图"]
LEAK_VOCAB = ["向左", "向右", "基本没动", "上行", "下行", "只有单音", "只有和弦", "都有", "穿指", "跨指",
              "落下", "抬起", "按键保持", "空中保持", "移动", "大三和弦", "小三和弦", "减三和弦", "增三和弦",
              "属七和弦", "大七和弦", "小七和弦", "半减七和弦", "减七和弦", "度", "低音方向", "高音方向"]


# ------------------------------------------------------------------------------------- prompt

def load_prompt(path):
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    sec = {parts[i].strip(): parts[i + 1].strip() for i in range(1, len(parts), 2)}
    shots = []
    i = 1
    while f"EXAMPLE {i} INPUT" in sec:
        shots.append((sec[f"EXAMPLE {i} INPUT"], sec[f"EXAMPLE {i} OUTPUT"]))
        i += 1
    return sec["SYSTEM"], shots


def card_for_llm(card):
    c = {k: v for k, v in card.items() if k not in ("frames", "clip_id", "video_id", "state_names")}
    c["frames"] = [f"t = {f['t']:.2f} 秒" for f in card["frames"]]
    return json.dumps(c, ensure_ascii=False)


def img_data(path, max_side):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_messages(card, system, shots, release, max_side):
    msgs = [{"role": "system", "content": system}]
    for inp, out in shots:
        msgs += [{"role": "user", "content": "事实卡：\n" + inp}, {"role": "assistant", "content": out}]
    content = [{"type": "text", "text": "事实卡：\n" + card_for_llm(card)}]
    if card["frames"]:
        content.append({"type": "text", "text": f"下面是这个片段在题目区间内按时间顺序的 {len(card['frames'])} 张画面："})
        for i, f in enumerate(card["frames"]):
            content.append({"type": "text", "text": f"[t = {f['t']:.2f} 秒]"})
            content.append({"type": "image_url", "image_url": {"url": img_data(os.path.join(release, f["path"]), max_side)}})
    return msgs + [{"role": "user", "content": content}]


# ------------------------------------------------------------------------------------- checks

def parse_json(text):
    t = (text or "").strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        raise ValueError("输出里找不到 JSON 对象")
    return json.loads(m.group(0))


def keys(s):
    return set(KEY_RE.findall(s or ""))


def nums_wo_keys(s):
    return set(NUM_RE.findall(KEY_RE.sub(" ", s or "")))


def validate(card, out):
    """Returns a list of problems (empty = accepted)."""
    errs = []
    for k in ("stem", "explanation", "visual_notes"):
        if not isinstance(out.get(k), str):
            errs.append(f"缺少字符串字段 {k}")
    if errs:
        return errs
    stem, expl, vis = out["stem"].strip(), out["explanation"].strip(), out["visual_notes"].strip()
    tpl, disp = card["template_stem"], card["answer"]["display"]
    # ---- stem
    n_ph = stem.count(PH)
    if card["needs_placeholder"] and n_ph != 1:
        errs.append(f"题干里必须恰好出现一次{PH}（现在 {n_ph} 次）")
    if not card["needs_placeholder"] and n_ph:
        errs.append(f"这道题不需要{PH}")
    body = stem.replace(PH, "")
    if card["task"] == "T5" and "切点处结束" in body:
        errs.append("预测题的题干不要再写“视频在切点处结束”，程序会自动加")
    if not 6 <= len(body) <= 100:
        errs.append(f"题干长度 {len(body)} 字，应在 6–100 字")
    if not re.search(r"[？?](（[^（）]*）)?$", stem):
        errs.append("题干要以问号结尾（补充说明放在问号前）")
    if "{" in stem or "}" in stem or re.search(r"(^|\n)\s*[A-E][.．、]", stem):
        errs.append("题干里不能有花括号或选项")
    if keys(body) != keys(tpl):
        errs.append(f"题干里的键名必须和原题一致：应为 {sorted(keys(tpl))}，现在是 {sorted(keys(body))}")
    extra_nums = nums_wo_keys(body) - nums_wo_keys(tpl)
    if extra_nums:
        errs.append(f"题干里出现了原题没有的数字 {sorted(extra_nums)}（区间要用{PH}表示）")
    for w in HANDS:
        if w in tpl and w not in body:
            errs.append(f"题干必须保留“{w}”")
        if w not in tpl and w in body and any(h in tpl for h in HANDS):
            errs.append(f"题干不能改问“{w}”")
    for w in FINGERS:
        if w in tpl and w not in body:
            errs.append(f"题干必须保留“{w}”")
        if w not in tpl and w in body:
            errs.append(f"题干里不能出现“{w}”（会泄露答案或改变问题）")
    if disp and disp not in tpl and disp in body:
        errs.append("题干泄露了答案")
    lists_all_states = all(x in body for x in ("落下", "抬起", "按键保持", "空中保持", "移动"))
    for w in LEAK_VOCAB:
        if lists_all_states and w in ("落下", "抬起", "按键保持", "空中保持", "移动"):
            continue
        if w in disp and w not in tpl and w in body and w not in ("度",):
            errs.append(f"题干里的“{w}”泄露了答案")
    # ---- explanation
    if not 10 <= len(expl) <= 400:
        errs.append(f"解析长度 {len(expl)} 字，应在 10–400 字")
    if not re.search(re.escape("所以答案是" + disp) + r"\s*[。.]?\s*$", expl):
        errs.append(f"解析最后一句必须是“所以答案是{disp}。”")
    long_t = [x for x in SEC_RE.findall(expl) if "." in x and len(x.split(".")[1]) > 1 and x not in disp]
    if long_t:
        errs.append(f"时刻统一写成“约 X.X 秒”，保留一位小数：{long_t}")
    allowed = keys(tpl) | {e["key"] for e in card["events"]} | keys(" ".join(card["answer"].get("options") or [])) | keys(disp)
    bad_keys = keys(expl) - allowed
    if bad_keys:
        errs.append(f"解析提到了事实卡里没有的键 {sorted(bad_keys)}")
    hi = card["clip_length_s"] + ((card.get("predict") or {}).get("horizon") or 0) + 0.6
    bad_t = [x for x in SEC_RE.findall(expl) if float(x) > hi and x not in nums_wo_keys(tpl + disp)]
    if bad_t:
        errs.append(f"解析里的时刻超出片段范围：{bad_t}")
    known = [e[k] for e in card["events"] for k in ("press_s", "release_s")]
    for k in ("interval", "predict"):
        known += [float(v) for v in (card.get(k) or {}).values() if isinstance(v, (int, float))]
    fv = card["answer"].get("fill_value")
    known += [float(x) for x in (fv if isinstance(fv, list) else [fv]) if isinstance(x, (int, float))]
    known += [float(x) for x in NUM_RE.findall(tpl + disp)]
    off = [x for x in SEC_RE.findall(expl) if not any(abs(float(x) - t) <= 0.15 for t in known)]
    if off:
        errs.append(f"解析里的时刻 {off} 秒对不上任何一次按键的按下或松开时刻，不要自己推算时刻")
    # ---- no references to where the information came from (SVBench rule)
    for w in SOURCE_WORDS:
        if w in stem or w in expl or w in vis:
            errs.append(f"不要提到信息来源“{w}”，要描述成视频里发生的事")
    if re.search(r"第\s*\d+\s*(至|到|-|–)?\s*\d*\s*张", stem + expl + vis):
        errs.append("不要写“第几张画面”，用“开头 / 中段 / 结尾”或秒数指代时刻")
    if re.search(r"[a-z_]+\s*=\s*-?\d", expl):
        errs.append("解析里不要写测量字段的原始数值（如 span_white = 6.16），要说成“约 6 个白键宽”")
    # ---- visual notes
    if vis and not card["frames"]:
        errs.append("没有附画面时 visual_notes 必须为空")
    if len(vis) > 150:
        errs.append("visual_notes 太长")
    return errs


# ------------------------------------------------------------------------------------- mock

def mock_reply(card, attempt, bad_rate, rng):
    """Deterministic stand-in for an LLM: sometimes wrong on the first try, to exercise the retry loop."""
    stem = (PH + "，" if card["needs_placeholder"] else "") + card["template_stem"]
    expl = f"视频里与这道题相关的按键共有 {len(card['events'])} 次。所以答案是{card['answer']['display']}。"
    out = {"stem": stem, "explanation": expl, "visual_notes": ""}
    if attempt == 0 and rng.random() < bad_rate:
        kind = rng.choice(["leak", "placeholder", "newkey", "noanswer"])
        if kind == "leak":
            out["stem"] = stem.rstrip("？?") + f"（提示：{card['answer']['display']}）？"
        elif kind == "placeholder":
            out["stem"] = stem.replace(PH, "第 1.0 秒到第 2.0 秒之间")
        elif kind == "newkey":
            out["explanation"] = expl + "随后还弹了 C8。"
        else:
            out["explanation"] = "这题很简单。"
    return json.dumps(out, ensure_ascii=False)


# ------------------------------------------------------------------------------------- run

def cmd_run(a):
    import random
    import requests
    system, shots = load_prompt(a.prompt)
    version = os.path.basename(a.prompt)
    cards = [json.loads(l) for l in open(a.cards, encoding="utf-8")]
    if a.tasks:
        keep = set(a.tasks.split(","))
        cards = [c for c in cards if c["task"] in keep]
    if a.sample:   # a few cards from every subtype, for trying out a model / prompt
        by = {}
        for c in cards:
            by.setdefault((c["task"], c["subtype"]), []).append(c)
        rs = __import__("random").Random(a.seed)
        cards = [c for k in sorted(by) for c in rs.sample(by[k], min(a.sample, len(by[k])))]
    if a.limit:
        cards = cards[: a.limit]
    if a.show:
        for c in cards[: a.show]:
            m = build_messages(c, system, shots, a.release, a.max_side)
            last = m[-1]["content"]
            n_img = sum(1 for x in last if x["type"] == "image_url")
            print("=" * 80, f"\n{c['fact_id']} {c['task']} {c['subtype']}  few-shot {len(shots)} 轮，图片 {n_img} 张")
            print(last[0]["text"])
        print("=" * 80, f"\nsystem prompt {len(system)} 字；共 {len(cards)} 张卡，其中 {sum(bool(c['frames']) for c in cards)} 张带图")
        return
    done = set()
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            r = json.loads(l)
            if r.get("ok") or r.get("final"):
                done.add(r["fact_id"])
    todo = [c for c in cards if c["fact_id"] not in done]
    print(f"{len(cards)} cards, {len(done)} done, {len(todo)} to run" + ("  [mock]" if a.mock else ""))
    url = a.base_url.rstrip("/") + "/chat/completions" if a.base_url else None
    head = {"Authorization": f"Bearer {a.api_key}"}
    lock = threading.Lock()
    rng = random.Random(a.seed)

    def call(msgs, card, attempt):
        if a.mock:
            return mock_reply(card, attempt, a.mock_bad, rng), None
        body = {"model": a.model, "messages": msgs, "temperature": a.temperature, "max_tokens": a.max_tokens}
        if a.json_mode:
            body["response_format"] = {"type": "json_object"}
        for k in range(4):
            r = requests.post(url, json=body, headers=head, timeout=a.timeout)
            if r.status_code in (429, 500, 502, 503) and k < 3:
                time.sleep(5 * (k + 1))
                continue
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
            j = r.json()
            return j["choices"][0]["message"].get("content") or "", j.get("usage")

    def work(card):
        msgs = build_messages(card, system, shots, a.release, a.max_side)
        rec = dict(fact_id=card["fact_id"], task=card["task"], subtype=card["subtype"], model=a.model if not a.mock else "mock",
                   prompt=version, attempts=[], ok=False)
        usage = []
        try:
            for attempt in range(a.retries + 1):
                raw, u = call(msgs, card, attempt)
                usage.append(u)
                try:
                    out = parse_json(raw)
                    errs = validate(card, out)
                except (ValueError, json.JSONDecodeError) as e:
                    out, errs = None, [f"不是合法的 JSON：{e}"]
                rec["attempts"].append(dict(raw=raw, errors=errs))
                if not errs:
                    rec.update(ok=True, stem=out["stem"].strip(), explanation=out["explanation"].strip(),
                               visual_notes=out["visual_notes"].strip())
                    break
                msgs = msgs + [{"role": "assistant", "content": raw},
                               {"role": "user", "content": "上一次的输出没有通过检查，问题如下：\n- " + "\n- ".join(errs) +
                                "\n请按要求重新输出完整的 JSON。"}]
            rec["final"] = True
        except Exception as e:  # noqa: BLE001 - network etc.: logged, retried on the next run
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["usage"] = usage
        with lock, open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    n_ok = n_fail = n_err = n_retry = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, c) for c in todo]), 1):
            r = fut.result()
            n_ok += r["ok"]
            n_err += "error" in r
            n_fail += (not r["ok"]) and "error" not in r
            n_retry += r["ok"] and len(r["attempts"]) > 1
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  通过 {n_ok}（其中重试后通过 {n_retry}）  多次不通过 {n_fail}  请求出错 {n_err}")
    if n_err:
        last = [json.loads(l) for l in open(a.out, encoding="utf-8")]
        msg = next((r["error"] for r in reversed(last) if r.get("error")), "")
        print(f"有请求出错（最近一次：{msg}）；重新运行同一条命令会补跑")


# ------------------------------------------------------------------------------------- judge

JUDGE_KEYS = ("faithful", "clear", "unique", "no_leak", "natural", "overall")


def cmd_judge(a):
    """Second LLM pass (SVBench-style quality scoring): score every accepted expansion on 5 dimensions."""
    import random
    import requests
    system = load_prompt_system(a.prompt)
    cards = {json.loads(l)["fact_id"]: json.loads(l) for l in open(a.cards, encoding="utf-8")}
    exps = [json.loads(l) for l in open(a.expansions, encoding="utf-8")]
    exps = [r for r in exps if r.get("ok") and r["fact_id"] in cards]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["fact_id"] for l in open(a.out, encoding="utf-8") if json.loads(l).get("scores")}
    todo = [r for r in exps if r["fact_id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(exps)} expansions, {len(done)} judged, {len(todo)} to judge" + ("  [mock]" if a.mock else ""))
    rng = random.Random(a.seed)
    url = a.base_url.rstrip("/") + "/chat/completions" if a.base_url else None
    lock = threading.Lock()

    def ask(r):
        card = {k: v for k, v in cards[r["fact_id"]].items() if k not in ("frames", "clip_id", "video_id", "state_names")}
        user = ("事实卡：\n" + json.dumps(card, ensure_ascii=False) + "\n\n编写者写出的题目：\n" +
                json.dumps({"stem": r["stem"], "explanation": r["explanation"], "visual_notes": r["visual_notes"]},
                           ensure_ascii=False))
        if a.mock:
            sc = {k: rng.choice([5, 8, 9, 10, 10, 10, 10]) for k in JUDGE_KEYS[:-1]}
            sc["overall"] = min(min(sc.values()) + 2, 10)
            return json.dumps({**sc, "comment": "模拟打分"}, ensure_ascii=False)
        body = {"model": a.model, "temperature": 0.0, "max_tokens": 300,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        r2 = requests.post(url, json=body, headers={"Authorization": f"Bearer {a.api_key}"}, timeout=a.timeout)
        if not r2.ok:
            raise RuntimeError(f"HTTP {r2.status_code}: {r2.text[:300]}")
        return r2.json()["choices"][0]["message"].get("content") or ""

    def work(r):
        rec = dict(fact_id=r["fact_id"], task=r["task"], subtype=r["subtype"], model=a.model or "mock",
                   prompt=os.path.basename(a.prompt))
        try:
            raw = ask(r)
            out = parse_json(raw)
            sc = {k: int(out[k]) for k in JUDGE_KEYS}
            rec.update(scores=sc, comment=str(out.get("comment", "")), raw=raw,
                       passed=sc["overall"] >= a.min_overall and min(sc.values()) >= a.min_each)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        with lock, open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    n_pass = n_flag = n_err = 0
    with ThreadPoolExecutor(a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(work, r) for r in todo]), 1):
            r = fut.result()
            n_err += "error" in r
            n_pass += bool(r.get("passed"))
            n_flag += ("scores" in r) and not r["passed"]
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  通过 {n_pass}  转人工 {n_flag}  出错 {n_err}")


def load_prompt_system(path):
    text = re.sub(r"<!--.*?-->", "", open(path, encoding="utf-8").read(), flags=re.S)
    return re.split(r"^## SYSTEM\s*$", text, flags=re.M)[1].split("\n## ")[0].strip()


# ------------------------------------------------------------------------------------- apply

def cmd_report(a):
    """Readable Markdown of reworded questions: template vs. new wording, answer, explanation, judge scores."""
    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8")]
    seen, rows = set(), []
    for q in qs:
        if q["fact_id"] in seen or not q.get("gen", {}).get("status", "").startswith("llm"):
            continue
        seen.add(q["fact_id"])
        rows.append(q)
    L = [f"# 扩写试跑结果（{len(rows)} 个事实）\n"]
    for q in rows:
        g = q["gen"]
        ans = q["answer"] if q["answer_type"] != "choice" else f"{q['answer']}（{q['options']['ABCDE'.index(q['answer'])]}）"
        L.append(f"## {q['qid']} · {q['task']} {q['subtype']} · {g['status']}\n")
        L.append(f"- 模板题干：{q['question_template'].splitlines()[0]}")
        L.append(f"- 扩写后：{q['question'].splitlines()[0]}")
        if q.get("options"):
            L.append("- 选项：" + "　".join(f"{'ABCDE'[i]}. {o}" for i, o in enumerate(q["options"])))
        L.append(f"- 标准答案：{ans}")
        L.append(f"- 解析：{q.get('explanation') or ''}")
        if q.get("visual_notes"):
            L.append(f"- 画面描述：{q['visual_notes']}")
        if g.get("judge"):
            L.append(f"- 质检打分：{g['judge']}；{g.get('judge_comment') or ''}")
        L.append(f"- 片段：`{q['clip_id']}`（模型：{g.get('model')}，提示：{g.get('prompt')}）\n")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"{len(rows)} facts -> {a.out}")


def cmd_apply(a):
    cards = {json.loads(l)["fact_id"]: json.loads(l) for l in open(a.cards, encoding="utf-8")}
    exp = {}
    for l in open(a.expansions, encoding="utf-8"):
        r = json.loads(l)
        if r.get("ok"):
            exp[r["fact_id"]] = r
    judged = {}
    if a.judgments and os.path.exists(a.judgments):
        for l in open(a.judgments, encoding="utf-8"):
            j = json.loads(l)
            if "scores" in j:
                judged[j["fact_id"]] = j
    n_new = n_tpl = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for l in open(os.path.join(a.release, "questions.jsonl"), encoding="utf-8"):
            q = json.loads(l)
            card, r = cards.get(q["fact_id"]), exp.get(q["fact_id"])
            old = q["question"]
            body = card["template_stem"] if card else None
            if r and body and body in old and not q["tags"].get("as_frame"):
                i = old.index(body)
                pre, suf = old[:i], old[i + len(body):]
                if card["needs_placeholder"]:
                    new = r["stem"].replace(PH, pre.rstrip("，,"), 1) + suf
                    new = re.sub(r"(之间|里)[内中里](?![央间部心段指])", r"\1", new)   # "之间内" -> "之间"
                else:
                    new = pre + r["stem"] + suf
                j = judged.get(q["fact_id"])
                status = "llm" if not j else ("llm_judged_ok" if j["passed"] else "llm_needs_review")
                q.update(question=new, question_template=old, explanation=r["explanation"],
                         visual_notes=r["visual_notes"], gen=dict(status=status, model=r["model"], prompt=r["prompt"],
                                                                  judge=(j or {}).get("scores"),
                                                                  judge_comment=(j or {}).get("comment")))
                n_new += 1
            else:
                q.update(question_template=old, explanation=None, visual_notes=None, gen=dict(status="template"))
                n_tpl += 1
            fo.write(json.dumps(q, ensure_ascii=False) + "\n")
    flagged = sorted({fid for fid, j in judged.items() if not j["passed"]})
    print(f"{n_new} questions reworded by the LLM, {n_tpl} kept the template wording -> {a.out}")
    if judged:
        print(f"质检：{len(judged)} 个事实打过分，{len(flagged)} 个转人工复核（gen.status = llm_needs_review）")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cards", required=True)
    r.add_argument("--release", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--prompt", default=os.path.join(os.path.dirname(__file__), "..", "prompts", "expand_qa_v4.md"))
    r.add_argument("--base-url", default="")
    r.add_argument("--model", default="")
    r.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    r.add_argument("--tasks", default="")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--sample", type=int, default=0, help="take N random cards from every subtype (for trying out)")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--retries", type=int, default=2)
    r.add_argument("--max-side", type=int, default=768)
    r.add_argument("--temperature", type=float, default=0.7)
    r.add_argument("--max-tokens", type=int, default=800)
    r.add_argument("--timeout", type=int, default=180)
    r.add_argument("--json-mode", action="store_true", help="send response_format=json_object (if the API supports it)")
    r.add_argument("--mock", action="store_true")
    r.add_argument("--mock-bad", type=float, default=0.3)
    r.add_argument("--show", type=int, default=0)
    r.add_argument("--seed", type=int, default=0)
    j = sub.add_parser("judge")
    j.add_argument("--cards", required=True)
    j.add_argument("--expansions", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--prompt", default=os.path.join(os.path.dirname(__file__), "..", "prompts", "judge_qa_v2.md"))
    j.add_argument("--base-url", default="")
    j.add_argument("--model", default="")
    j.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    j.add_argument("--min-overall", type=int, default=8)
    j.add_argument("--min-each", type=int, default=6)
    j.add_argument("--limit", type=int, default=0)
    j.add_argument("--workers", type=int, default=4)
    j.add_argument("--timeout", type=int, default=120)
    j.add_argument("--mock", action="store_true")
    j.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("apply")
    p.add_argument("--release", required=True)
    p.add_argument("--cards", required=True)
    p.add_argument("--expansions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judgments", default="", help="output of `judge`; low-scoring facts are marked llm_needs_review")
    rp = sub.add_parser("report")
    rp.add_argument("--questions", required=True, help="output of apply")
    rp.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd in ("run", "judge") and not (a.mock or getattr(a, "show", 0)) and not (a.base_url and a.model):
        ap.error(f"{a.cmd} needs --base-url and --model (or --mock)")
    {"run": cmd_run, "judge": cmd_judge, "apply": cmd_apply, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
