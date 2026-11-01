"""Deterministic Bench question text for a fact (the scored version, spec v0.3.1). The LLM never writes these:
it only writes the SFT question / evidence-bearing answer, so the scored wording stays fixed and auditable.
Intervals are half-open [a, b): an event counts if its key-down time is >= a and < b."""
from __future__ import annotations

LETTERS = "ABCDEFG"
PRED = "画面在最后一帧（第 {t0:.3f} 秒）之后还会继续，这一帧就是“现在”。"


def range_text(r):
    if not r or r["type"] == "whole":
        return "这段视频里"
    if r["type"] == "timestamp":
        return f"从第 {r['start']:.3f} 秒到第 {r['end']:.3f} 秒（不含）之间"
    if r["type"] == "frame_point":
        return f"第 {r['frame']} 帧（第 {r['t']:.3f} 秒）时"
    raise ValueError(r)


def opts_text(fact):
    ch = fact["options"]["choices"]
    return "选项：" + "；".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(ch)) + "。把选项字母写在花括号里。"


def bench_question(fact):
    s, sub, rt = fact["slots"], fact["subtype"], range_text(fact.get("range"))
    pr = PRED.format(t0=fact["predict"]["t0"]) if fact.get("predict") else ""
    H = fact["predict"]["horizon"] if fact.get("predict") else None
    q = {
        "按键次数·整只手": f"{rt}，{s.get('hand')}一共发生了几次按键？以每组最早按下时刻起40毫秒内的键为一组，不链式延长；一组算一次。把数字写在花括号里。",
        "按键次数·指定手指": f"{rt}，{s.get('hand')}{s.get('finger')}一共按下了几次键？把数字写在花括号里。",
        "按键保持时长": f"{rt}，{s.get('key')} 这个键从按下到松开保持了多久？",
        "张开跨度": f"{rt}，{s.get('hand')}拇指指尖与小指指尖之间的横向距离，大约相当于多少个白键的宽度？",
        "手掌净位移": f"{rt}，{s.get('hand')}手掌中心从区间开始到结束的横向净位移，方向和幅度是什么？"
                     "（画面左右即琴键低音和高音方向）",
        "持键重叠": f"{rt}，{s.get('hand_b')}按下 {s.get('key_b')} 的那一刻，之前按下的 {s.get('key_a')} 是还按着，还是已经松开？"
                    "（这段视频里两个键各只按了一次）",
        "双手配合": f"{rt}，{s.get('hand_b')}按下 {s.get('key_b')} 的那一刻，{s.get('hand_a')}之前按下的 {s.get('key_a')} "
                    "是还按着，还是已经松开？（这段视频里两个键各只按了一次）",
        "先松开哪个键": f"{rt}，{s.get('key_a')} 和 {s.get('key_b')} 几乎同时按下（这段视频里两个键各只按了这一次），哪一个先松开？",
        "旋律最高点": f"这段视频里，{s.get('hand')}的连续单音第一次出现“先升后降”，最高的那个音是在第几秒按下的？"
                     "把时间（秒）写在花括号里。",
        "第一次三音以上同时按下": f"这段视频里，{s.get('hand')}第一次几乎同时按下三个或更多不同的键，是在第几秒？"
                              "把时间（秒）写在花括号里。",
        "第一次按进新的八度": f"这段视频里，{s.get('hand')}第一次按下 {s.get('octave')} 范围内的键是在第几秒？"
                           "把时间（秒）写在花括号里。",
        "第一次大幅横移": f"这段视频里，{s.get('hand')}第一次在一秒左右之内横向移动约一个八度或更远，这段移动从第几秒开始、"
                       "到第几秒结束？把两个时间（秒）写在花括号里，用逗号分隔。",
        "三音键序列": f"这段视频里，{s.get('hand')}连续依次按下 {'、'.join(s.get('keys', []))}，这一段从第几秒（第一个键按下）"
                     "到第几秒（最后一个键按下）？把两个时间（秒）写在花括号里，用逗号分隔。",
        "指定手指按指定键": f"这段视频里，{s.get('hand')}{s.get('finger')}按下 {s.get('key')} 是在第几秒？把时间（秒）写在花括号里。",
        "长按某个键": f"这段视频里，{s.get('key')} 有一次被持续按住至少 1 秒，这段按住从第几秒开始、到第几秒松开？"
                     "把两个时间（秒）写在花括号里，用逗号分隔。",
        "会不会发生新的按键": f"{pr}接下来 {H} 秒内，{s.get('hand')}会不会再发生一次按键（重复按同一个键也算）？",
        "某个键之后是否处于按下状态": f"{pr}{s.get('hand')}此刻按着 {s.get('key')}。{H} 秒之后的那一刻，{s.get('key')} "
                                   "是处于按下状态还是松开状态？（中途松开后又重新按下，也算按下状态）",
        "手掌横移方向": f"{pr}接下来 {H} 秒，{s.get('hand')}手掌中心相对现在的横向净位移是什么？",
        "下一个键（选择）": f"{pr}{s.get('hand')}接下来 {H} 秒内按下的第一个键是哪个？",
        "下一个键（填空）": f"{pr}{s.get('hand')}接下来 {H} 秒内按下的第一个键是哪个？把键名写在花括号里，例如 {{G#3}}。",
        "下一根手指": f"{pr}{s.get('hand')}接下来 {H} 秒内按下下一个键，会用哪根手指？",
    }[sub]
    if fact.get("options"):
        q += opts_text(fact)
    return q


def bench_answer(fact):
    a, t = fact["answer"], fact["answer_type"]
    if t == "time_span":
        return "{" + f"{a[0]:.2f}, {a[1]:.2f}" + "}"
    if t == "time_point":
        return "{" + f"{a:.2f}" + "}"
    return "{" + str(a) + "}"


def answer_text(fact):
    """Human-readable answer (option text where there are options)."""
    if fact.get("options"):
        o = fact["options"]
        return f"{o['letter']}. {o['choices'][LETTERS.index(o['letter'])]}"
    a = fact["answer"]
    if fact["answer_type"] == "time_span":
        return f"第 {a[0]:.2f} 秒到第 {a[1]:.2f} 秒"
    if fact["answer_type"] == "time_point":
        return f"第 {a:.2f} 秒"
    return str(a)
