"""Offline-only P0 restoration registry. Never imports/constructs an API client.

Consumes frozen upstream performance/result files; writes only a new output directory.
The 11 gold final-test recordings are explicitly refused. Synthetic fixtures test gold paths.
"""
import argparse
import collections
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from batch_sft import atomic_json, digest, file_hash, frame_pack


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


rules = module("p0_rules", ROOT / "pmfinger/p0_rules.py")
state = rules._states
templates = module("p0_templates", ROOT / "pmfinger/p0_templates.py")
PROFILE = "vam-p0-restore-v1"
T3 = {"按键次数·整只手": "t3_count_hand", "按键次数·指定手指": "t3_count_finger",
      "按键保持时长": "t3_hold", "张开跨度": "t3_span", "手掌净位移": "t3_move",
      "持键重叠": "t3_overlap", "双手配合": "t3_overlap", "先松开哪个键": "t3_release_order"}
T4 = ["旋律最高点", "第一次三音以上同时按下", "第一次按进新的八度", "第一次大幅横移",
      "三音键序列", "指定手指按指定键", "长按某个键"]
T5 = ["会不会发生新的按键", "某个键之后是否处于按下状态", "手掌横移方向",
      "下一个键（选择）", "下一个键（填空）", "下一根手指"]
REGISTRY = {"T2/哪根手指": "finger_at_onset/key_unique"}
REGISTRY.update({"T3/"+s: "onset_group_count_40ms" if s == "按键次数·整只手" else s for s in T3})
REGISTRY.update({"T4/"+s: "third_onset_interval" if s == "三音键序列" else s for s in T4})
REGISTRY.update({"T5/"+s: s for s in T5})
FINGER = set(rules.FINGER_SUBTYPES) | {"哪根手指"}
GEOMETRY = {"张开跨度", "手掌净位移", "第一次大幅横移", "手掌横移方向"}


class GateError(ValueError):
    pass


def groups_in_window(P, h, a, b):
    full = P.groups(np.flatnonzero(P.hand == h), .04)
    selected = []
    for group in full:
        inside = [n for n in group if a <= P.on[n] < b]
        if inside and len(inside) != len(group):
            raise GateError("window_splits_complete_onset_group")
        if inside:
            selected.append(group)
    return selected


def retrospective_gate(P, h, a, b, frames, group_sensitive=True):
    if len(P.hand_unknown_in(a, b)):
        raise GateError("unknown_hand_can_change_answer")
    g = rules.Gen(P, random.Random(0))
    groups = groups_in_window(P, h, a, b)
    if not g.countable(groups, frames):
        raise GateError("press_not_visible_or_not_countable")
    # Ambiguous 40 ms group membership cannot be resolved at native frame precision.
    if group_sensitive:
        from batch_sft import check_event_visibility
        check_event_visibility(P.on, P.off, P.pitch, P.on_s, groups, frames, P.fps, True)
    return groups


def validate_fact(P, f):
    frames = [x["video_frame"] for x in f["frames"]]
    if (not frames or frames != sorted(set(frames)) or frames[0] < 0 or frames[-1] >= P.T
            or any(abs(x["t"] - (x["video_frame"] - frames[0])/P.fps) > 1e-9 for x in f["frames"])):
        raise GateError("delivery_timeline_invalid")
    h = state.HAND_ZH.index(f["slots"]["hand"]) if "hand" in f["slots"] else None
    if f["task"] != "T5" and f["subtype"] not in GEOMETRY:
        r = f.get("range")
        a, b = ((frames[0] + round(r["start"]*P.fps), frames[0] + round(r["end"]*P.fps))
                if r and r["type"] == "timestamp" else (frames[0], frames[-1]+1))
        for hand in ([h] if h is not None else [0,1]):
            retrospective_gate(P, hand, a, b, frames)
    if f["subtype"] in FINGER:
        f["needs_human"] = True
        if not any("finger" in a for a in f["atoms"]) and f["task"] != "T5":
            raise GateError("finger_requires_manual_GT")
        if f["subtype"] == "下一根手指":
            cut = frames[-1]
            groups = P.groups(np.flatnonzero(P.hand == h), .04)
            future = [g for g in groups if any(P.on[n] > cut for n in g)]
            if not future or len(future[0]) != 1 or P.finger[future[0][0]] < 0:
                raise GateError("finger_requires_manual_GT")
        # The generator may expose only annotated onset fingers, never a holder identity.
        if any("finger" in a and a.get("finger_source") != "人工金标" for a in f["atoms"]):
            raise GateError("finger_requires_manual_GT")
    if f["task"] == "T5":
        retrospective_gate(P, h, frames[0], frames[-1]+1, frames, False)
        cut = f["predict"]["t0"]
        if abs(cut - f["frames"][-1]["t"]) > 1e-9 or any(a["t"] > cut+1e-9 for a in f["atoms"]):
            raise GateError("future_information_in_observation")
    return f


def t2_finger(g, c0, c1):
    P = g.P
    cand = [n for n in g.once_in_clip(c0,c1) if P.finger[n] >= 0 and P.hand[n] >= 0]
    if not cand:
        return None
    n = g.rnd.choice(cand)
    f = int(P.finger[n])
    return g.fact("T2", "哪根手指", "L5", c0,c1,8,
                  dict(hand=state.HAND_ZH[P.hand[n]], key=state.pitch_name(P.pitch[n])), "choice", rules.LETTERS[f],
                  options=dict(choices=state.FINGER_ZH.copy(), letter=rules.LETTERS[f]), rng={"type":"whole"})


def freeze(P, f, split, source_hash):
    f = validate_fact(P, f)
    public = {k: f[k] for k in ("task","subtype","level","video_id","clip","frames","range","predict","slots","answer_type")}
    public["semantic_id"] = PROFILE + "/" + REGISTRY[f["task"]+"/"+f["subtype"]]
    public["split"] = split
    public["question"] = ("这段视频里，"+f["slots"]["hand"]+"按下 "+f["slots"]["key"]+" 的那一刻用的是哪根手指？"+templates.opts_text(f)
                          if f["task"] == "T2" else templates.bench_question(f))
    if f.get("options"):
        public["options"] = f["options"]["choices"]
    # No forecast outcome/difficulty/dx/target event is part of B/D evidence.
    atoms = f["atoms"]
    target = dict(answer=f["answer"], difficulty=f["difficulty"])
    if f["task"] == "T5":
        cut_native = f["frames"][-1]["video_frame"]
        target_native = cut_native + round(f["predict"]["horizon"]*P.fps)
        target.update(cut_native=cut_native, target_native=target_native,
                      events=[dict(note=int(n),hand=int(P.hand[n]),key=state.pitch_name(P.pitch[n]),
                                   on=int(P.on[n]),off=int(P.off[n]),finger_at_onset=int(P.finger[n]))
                              for n in range(len(P.pitch)) if P.on[n] <= target_native+round(.05*P.fps)
                              and P.off[n] > cut_native])
    private = dict(private_target=target)
    fact_hash = digest([public, private, atoms, source_hash])
    return dict(qid=PROFILE+"-"+fact_hash[:20], fact_hash=fact_hash, source_hash=source_hash,
                question=public, atoms=atoms, private_target=target,
                needs_human=f["subtype"] in FINGER,
                status="needs_human_GT_review" if f["subtype"] in FINGER else "offline_candidate")


def verify_frozen(row):
    expected = digest([row["question"], dict(private_target=row["private_target"]), row["atoms"], row["source_hash"]])
    if (row["fact_hash"] != expected or row["qid"] != PROFILE+"-"+expected[:20]
            or row["needs_human"] != (row["question"]["subtype"] in FINGER)):
        raise GateError("frozen_fact_changed")


def approved(row, approvals):
    """Stale approvals cannot authorize changed facts. No caller may assume a TSV means approval."""
    try:
        verify_frozen(row)
    except GateError:
        return False
    return any(a.get("qid") == row["qid"] and a.get("fact_hash") == row["fact_hash"]
               and a.get("decision") == "approve" and bool(a.get("reviewer", "").strip()) for a in approvals)


def writer_input(row):
    verify_frozen(row)
    return dict(question=row["question"], atoms=row["atoms"])


def compose_sft(row, observation, clauses, approvals=()):
    verify_frozen(row)
    if row["needs_human"] and not approved(row, approvals):
        raise GateError("human_approval_missing_or_stale_fact_hash")
    by_id = {a["id"]: a for a in row["atoms"]}
    if observation != "交付画面中可见键盘，所问手部所在区域清晰可辨。":
        raise GateError("invalid_observation_section")
    if not clauses:
        raise GateError("analysis_empty")
    texts = []
    for c in clauses:
        a = by_id[c["atom_id"]]
        expected = atom_sentence(a)
        if c.get("text") != expected:
            raise GateError("clause_not_equal_to_cited_fact")
        texts.append(expected)
    answer = row["private_target"]["answer"]
    if row["question"]["answer_type"] == "choice" and (not isinstance(answer,str) or answer not in rules.LETTERS[:len(row["question"]["options"])]):
        raise GateError("invalid_program_answer_letter")
    if isinstance(answer,list):
        answer = ", ".join(f"{x:.3f}" for x in answer)
    return "【观察】"+observation+"【分析】"+"".join(texts)+"所以答案是{"+str(answer)+"}"


def atom_sentence(a):
    # Exact relation, including finger-at-onset, is bound to one atom; no token-whitelist shortcut.
    finger = "（按下时使用"+a["finger"]+"）" if "finger" in a else ""
    x = f"，横坐标为{a['x']:.2f}个白键宽" if "x" in a else ""
    return f"约{a['t']:.3f}秒，{a['subject']}{a['predicate']}{a['object']}{finger}{x}。"


def scan(P, split, source_hash, quota=4, stride_s=.5):
    stats = {k: dict(scanned=0, eligible=0, selected=0, rejected={}) for k in REGISTRY}
    g = rules.Gen(P, random.Random(int(source_hash[:16],16)))
    rows, seen, intervals = [], set(), collections.defaultdict(list)
    def accept(key, f, c0, c1):
        stat = stats[key]
        stat["scanned"] += 1
        try:
            if f is None or f["task"]+"/"+f["subtype"] != key:
                raise GateError("no_manual_GT_in_video" if key.split("/",1)[1] in FINGER and not (P.finger >= 0).any()
                                else "no_eligible_fact_under_spec_gates")
            row = freeze(P,f,split,source_hash)
            stat["eligible"] += 1
            if row["qid"] in seen:
                raise GateError("exact_duplicate")
            if any(max(c0,a)<min(c1,b) for a,b in intervals[key]):
                raise GateError("overlapping_same_semantic_window")
            if stat["selected"] >= quota:
                raise GateError("video_subtype_quota")
            stat["selected"] += 1
            rows.append(row)
            seen.add(row["qid"])
            intervals[key].append((c0,c1))
        except (ValueError,IndexError,StopIteration) as e:
            reason = str(e) or type(e).__name__
            stat["rejected"][reason] = stat["rejected"].get(reason,0)+1
    step = max(1,round(P.fps*stride_s))
    for c0 in range(0, P.T-round(2.6*P.fps), step):
        c1 = c0+round(5*P.fps)
        if c1 <= P.T:
            for sub, method in T3.items():
                accept("T3/"+sub,getattr(g,method)(c0,c1),c0,c1)
            accept("T2/哪根手指",t2_finger(g,c0,c1),c0,c1)
        c1 = c0+round(6*P.fps)
        if c1 <= P.T:
            found = {f["subtype"]:f for f in g.t4(c0,c1)}
            for sub in T4:
                accept("T4/"+sub,found.get(sub),c0,c1)
        cut = c0+round(2*P.fps)
        for h in (0,1):
            found = collections.defaultdict(list)
            for f in g.t5(cut,h):
                found[f["subtype"]].append(f)
            for sub in T5:
                for f in found.get(sub,[None]):
                    accept("T5/"+sub,f,c0,cut+1)
    return rows,stats


def render_page(out, rows):
    cards=[]
    for row in rows:
        q=row["question"]
        images="".join('<img loading="lazy" src="'+html.escape(m["path"])+ '">' for m in row["media"]["packs"])
        cards.append('<article><h2>'+html.escape(q["task"]+" / "+q["subtype"])+ '</h2><p>'+html.escape(q["question"])+
                     '</p><small>'+html.escape(row["qid"]+" / "+row["status"])+ '</small><details><summary>程序真值和证据（离线，未调 API）</summary><pre>'+html.escape(json.dumps(dict(private_target=row["private_target"],atoms=row["atoms"]),ensure_ascii=False,indent=2))+ '</pre></details><div>'+images+'</div></article>')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>P0 两段开发视频离线样题</title><style>body{font:17px system-ui;max-width:1000px;margin:30px auto;background:#f5f6f8}article{background:white;padding:24px;margin:20px 0}img{max-width:100%}pre{white-space:pre-wrap}small{color:#666}</style><h1>P0 离线候选样题</h1><p>未调用 API；候选不等于验收通过；指法必须人工 GT 与逐题批准。</p>'+''.join(cards),encoding='utf-8')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,help='read-only release_v2 root')
    ap.add_argument('--out',required=True,help='new empty isolated directory')
    ap.add_argument('--videos',nargs=2,required=True)
    ap.add_argument('--quota',type=int,default=2)
    a=ap.parse_args()
    source,out=Path(a.source).resolve(),Path(a.out).resolve()
    if out.exists() or out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise SystemExit('out must be a new isolated directory')
    plan=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    by_id={r['video_id']:r for r in plan['records']}
    if len(set(a.videos))!=2 or any(v in plan['gold_ids'] or by_id[v]['split']!='dev' or by_id[v]['missing'] for v in a.videos):
        raise SystemExit('exactly two eligible DEV videos required; final-test/train refused')
    out.mkdir(parents=True)
    # Read-only link for the existing shared frame renderer, no exporter or decoder run.
    (out/'perf').symlink_to(source/'perf',target_is_directory=True)
    allrows=[]
    stats={}
    for vid in a.videos:
        files=[source/'perf'/(vid+s) for s in ('.npz','.meta.json','.skeleton.npz')]+[source/'fing'/(vid+'.fing.npz')]
        source_hash=digest({str(p.relative_to(source)):file_hash(p) for p in files})
        P=state.load_raw(files[0],files[-1])
        rows,counts=scan(P,'dev',source_hash,a.quota)
        stats[vid]=counts
        d=SimpleNamespace(vid=vid,fps=P.fps,perf=SimpleNamespace(video_path=P.video_path))
        for row in rows:
            q=row['question']
            frames=q['frames']
            compatible=dict(task=q['task'],clip_id=row['qid'],frame_fps=q['clip']['fps_out'], delivery_frames=[f['video_frame'] for f in frames],
                            clip=dict(start_frame=frames[0]['video_frame'],end_frame=frames[-1]['video_frame']+1),
                            predict=dict(cut=q['predict']['t0']) if q['predict'] else None)
            try:
                row['media']=frame_pack(out,compatible,d)
                expected=[f['video_frame'] for f in frames]
                actual=[f['native_frame'] for f in row['media']['frames']]
                if expected!=actual:
                    raise GateError('actual_delivery_grid_mismatch')
            except Exception as e:
                row.update(status='delivery_failed',error=str(e))
            atomic_json(out/'results'/(row['qid']+'.json'),row)
            allrows.append(row)
        atomic_json(out/'counts'/(vid+'.json'),counts)
    atomic_json(out/'dry_run.json',dict(profile=PROFILE,api_calls=0,videos=a.videos,counts=stats,
                candidates=len(allrows),delivered=sum('media' in r and r['status']!='delivery_failed' for r in allrows)))
    render_page(out,[r for r in allrows if r['status']!='delivery_failed'])
    print(json.dumps(dict(out=str(out),candidates=len(allrows),api_calls=0),ensure_ascii=False))


if __name__=='__main__':
    main()
