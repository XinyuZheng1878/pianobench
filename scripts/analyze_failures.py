"""Failure analysis of the fingering pipeline against labelled notes.

Run it on the VALIDATION split (auto labels), never on the sealed human-GT pieces: findings here
drive design changes, and the GT set must stay untouched for the final number.

For every labelled note it records the outcome and the evidence behind it, then attributes each
error to a mechanism:
  abstain/<status>   no label produced (low_conf, no_view, suspect, overfull)
  hand               wrong hand
  geometry           the true finger's tip was not near the key (tracking/projection failure)
  ambiguous          true and nearest other finger almost equally close (margin < 0.2 key)
  prior_override     raw evidence picked the right finger, sequence decoding changed it
  evidence           another finger was clearly closer to the key than the true finger
It also measures release tracking against PianoVAM's real key-release times, with MIDI offsets
pushed back by --release-extra seconds to mimic pedal-sustained transcription offsets (as in PM10M).

  python -B scripts/analyze_failures.py --inputs 'split/validation/*.npz' --prior prior.npz \
      --set sigma_white=0.4 w_seq=1 conf_thresh=0.7 --out analysis_val
"""
import argparse
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pmfinger import STATUS, Config, FingerPrior, apply_overrides, load_performance, run_fingering  # noqa: E402
from pmfinger.evidence import finger_evidence  # noqa: E402
from pmfinger.keys import IS_BLACK, pitch_name  # noqa: E402
from pmfinger.preprocess import clean_tips  # noqa: E402
from pmfinger.schema import check_output_path  # noqa: E402

GEOM_FAR = 0.75      # true fingertip farther than this (white keys) from the key -> geometry failure
AMBIG = 0.2          # nearest-vs-true distance gap below this -> ambiguous


def analyze_one(job):
    path, prior_path, overrides, extra = job
    cfg = apply_overrides(Config(), overrides)
    prior = FingerPrior.load(prior_path) if prior_path else FingerPrior.default()
    perf = load_performance(path)
    if perf.label_hand is None or perf.label_finger is None:
        return []
    true_off = perf.offset.copy()
    perf.offset = perf.offset + extra
    res = run_fingering(perf, prior, cfg)
    tips, ok = clean_tips(perf.tips, perf.valid, cfg)
    on_f = res['onset_frame']
    _, s, avail, resid = finger_evidence(tips, ok, perf.pitch, on_f, perf.fps, res['bias'], cfg)
    dist = np.abs(resid - res['bias'][None])                     # (N,2,5) tip-to-key distance after bias
    lh, lf = perf.label_hand, perf.label_finger
    T = tips.shape[0]

    chord = np.array([int(np.sum(np.abs(perf.onset - t) <= cfg.chord_tol_s)) for t in perf.onset])
    ioi = np.full(perf.N, np.nan)
    for h in (0, 1):
        idx = np.flatnonzero(lh == h)
        if len(idx) > 1:
            ioi[idx[1:]] = np.diff(perf.onset[idx])

    rows = []
    for n in np.flatnonzero((lh >= 0) & (lf >= 0)):
        h, f = int(lh[n]), int(lf[n])
        hp, fp, st = int(res['hand_pred'][n]), int(res['finger_pred'][n]), int(res['status'][n])
        t = int(np.clip(on_f[n], 0, T - 1))
        gap = np.nan
        if ok[t, 0] and ok[t, 1]:
            gap = float(np.min(np.abs(tips[t, 0, :, 0][:, None] - tips[t, 1, :, 0][None, :])))
        d = dist[n, h]
        has_ev = bool(avail[n, h])
        ev_f = int(np.argmin(d)) if has_ev else -1
        others = np.delete(d, f)
        margin = float(others.min() - d[f]) if has_ev else np.nan    # >0: true finger is the closest
        if st != 0:
            cause = f'abstain/{STATUS[st]}'
        elif hp != h:
            cause = 'hand'
        elif fp == f:
            cause = 'correct'
        elif not has_ev or d[f] > GEOM_FAR:
            cause = 'geometry'
        elif abs(margin) < AMBIG:
            cause = 'ambiguous'
        elif ev_f == f:
            cause = 'prior_override'
        else:
            cause = 'evidence'
        rel_err = np.nan
        if res['release_frame'][n] >= 0 and hp == h and fp == f:
            rel_err = float(res['release_s'][n] - true_off[n])
        rows.append(dict(
            video=perf.video_id, note=int(n), pitch=int(perf.pitch[n]), key=pitch_name(perf.pitch[n]),
            black=int(IS_BLACK[perf.pitch[n]]), onset=round(float(perf.onset[n]), 3),
            hand=h, finger=f + 1, hand_pred=hp, finger_pred=fp + 1, status=STATUS[st],
            conf=round(float(res['conf'][n]), 4), hand_conf=round(float(res['hand_conf'][n]), 4),
            correct_if_forced=int(hp == h and fp == f), cause=cause,
            dist_true=round(float(d[f]), 3) if has_ev else '', margin=round(margin, 3) if has_ev else '',
            evidence_pick=ev_f + 1, chord=int(chord[n]), ioi=round(float(ioi[n]), 3) if np.isfinite(ioi[n]) else '',
            hand_gap=round(gap, 3) if np.isfinite(gap) else '', release_err=round(rel_err, 3) if np.isfinite(rel_err) else '',
            true_dur=round(float(true_off[n] - perf.onset[n]), 3), av_offset=int(res['av_offset']),
        ))
    return rows


# ----------------------------------------------------------------------------- aggregation

def acc_table(rows, key, bins=None, labels=None):
    """Accepted-output accuracy and coverage per group."""
    groups = defaultdict(list)
    for r in rows:
        v = r[key]
        if bins is not None:
            if v == '':
                g = 'n/a'
            else:
                i = int(np.searchsorted(bins, float(v), side='right'))
                g = labels[i]
        else:
            g = v
        groups[g].append(r)
    order = labels + ['n/a'] if labels else sorted(groups, key=str)
    out = []
    for g in order:
        if g not in groups:
            continue
        rs = groups[g]
        acc = [r for r in rs if r['status'] == 'ok']
        out.append((g, len(rs), len(acc) / len(rs),
                    np.mean([r['cause'] == 'correct' for r in acc]) if acc else np.nan,
                    np.mean([r['correct_if_forced'] for r in rs])))
    return out


def fmt_table(title, rows, head='分组'):
    s = [f'### {title}', '', f'| {head} | 音数 | 覆盖率 | 接受输出准确率 | 强制全答准确率 |', '|---|---|---|---|---|']
    for g, n, cov, acc, forced in rows:
        s.append(f'| {g} | {n} | {cov:.1%} | {acc:.1%} | {forced:.1%} |')
    return '\n'.join(s) + '\n'


def calibration(rows, nb=10):
    rs = [r for r in rows if r['status'] in ('ok', 'low_conf')]
    conf = np.array([r['conf'] for r in rs])
    hit = np.array([r['correct_if_forced'] for r in rs])
    edges = np.linspace(0, 1, nb + 1)
    lines, ece = [], 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b) if b < 1 else (conf >= a)
        if m.sum():
            ece += m.mean() * abs(conf[m].mean() - hit[m].mean())
            lines.append(f'| {a:.1f}–{b:.1f} | {int(m.sum())} | {conf[m].mean():.3f} | {hit[m].mean():.3f} |')
    return ece, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', required=True)
    ap.add_argument('--prior', default='')
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--release-extra', type=float, default=3.0,
                    help='seconds added to MIDI offsets so release must come from tracking (0 = true offsets as cap)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    check_output_path(args.out)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    files = sorted(glob.glob(args.inputs))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for rs in ex.map(analyze_one, [(f, args.prior, args.set, args.release_extra) for f in files]) for r in rs]
    with open(out / 'per_note.csv', 'w', newline='', encoding='utf-8-sig') as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    causes = Counter(r['cause'] for r in rows)
    acc_rows = [r for r in rows if r['status'] == 'ok']
    err_causes = Counter(r['cause'] for r in acc_rows if r['cause'] != 'correct')
    wrong = [r for r in acc_rows if r['cause'] not in ('correct', 'hand')]
    df = np.array([r['finger_pred'] - r['finger'] for r in wrong])
    ece, cal = calibration(rows)
    rel = np.array([r['release_err'] for r in rows if r['release_err'] != ''], dtype=float)
    abst = [r for r in rows if r['status'] != 'ok']

    md = [f'# 指法失败机制分析', '',
          f'- 输入：{len(files)} 段，{n} 个有标签的音；参数 {" ".join(args.set) or "默认"}；先验 {args.prior or "手写默认"}',
          f'- 注意：标签若来自 PianoVAM 自动标签（约 95.7% 准确），“错误”里有一部分其实是标签本身错了。', '',
          '## 1. 总体', '',
          f'- 覆盖率 {len(acc_rows) / n:.1%}，接受输出准确率 {causes["correct"] / max(len(acc_rows), 1):.1%}，'
          f'强制全答准确率 {np.mean([r["correct_if_forced"] for r in rows]):.1%}',
          f'- 置信度校准误差 ECE = {ece:.3f}（越接近 0 越好）', '',
          '## 2. 错误来源（只算被接受的输出）', '', '| 原因 | 个数 | 占错误比例 | 含义 |', '|---|---|---|---|']
    meaning = dict(hand='手分错', geometry='真实手指的指尖根本不在该键附近（跟踪/投影失败）',
                   ambiguous='真实手指和另一根手指离键几乎一样近', prior_override='证据本来选对了，被序列先验改错',
                   evidence='另一根手指明显更靠近该键')
    tot_err = sum(err_causes.values())
    for c, k in err_causes.most_common():
        md.append(f'| {c} | {k} | {k / max(tot_err, 1):.1%} | {meaning.get(c, "")} |')
    if len(df):
        md += ['', f'- 手指错误中相邻手指占 {np.mean(np.abs(df) == 1):.1%}；'
                   f'错向拇指侧 {int((df < 0).sum())} 个、错向小指侧 {int((df > 0).sum())} 个']
        conf_pairs = Counter((r['finger'], r['finger_pred']) for r in wrong).most_common(8)
        md.append('- 最常见的（真实→预测）手指混淆：' + '，'.join(f'{a}→{b}×{k}' for (a, b), k in conf_pairs))
    md += ['', '## 3. 放弃作答的音', '', '| 状态 | 个数 | 若强制作答的准确率 |', '|---|---|---|']
    for stt, k in Counter(r['status'] for r in abst).most_common():
        md.append(f'| {stt} | {k} | {np.mean([r["correct_if_forced"] for r in abst if r["status"] == stt]):.1%} |')
    md += ['', '## 4. 分组表现', '',
           fmt_table('按证据裕度（真实手指比最近的其他手指近多少，白键宽）',
                     acc_table(rows, 'margin', [-0.4, -0.1, 0.1, 0.4], ['<-0.4', '-0.4~-0.1', '-0.1~0.1', '0.1~0.4', '≥0.4'])),
           fmt_table('按真实手指离键距离（白键宽）', acc_table(rows, 'dist_true', [0.25, 0.5, 0.75], ['<0.25', '0.25~0.5', '0.5~0.75', '≥0.75'])),
           fmt_table('按真实手指', acc_table(rows, 'finger')),
           fmt_table('按黑白键（1=黑键）', acc_table(rows, 'black')),
           fmt_table('按同时发声的音数', acc_table(rows, 'chord', [1.5, 2.5, 3.5], ['1', '2', '3', '≥4'])),
           fmt_table('按与同手上一音的间隔（秒）', acc_table(rows, 'ioi', [0.1, 0.2, 0.5], ['<0.1', '0.1~0.2', '0.2~0.5', '≥0.5'])),
           fmt_table('按双手最近距离（白键宽，越小越可能交叉/重叠）', acc_table(rows, 'hand_gap', [1, 3, 6], ['<1', '1~3', '3~6', '≥6'])),
           fmt_table('按视频', acc_table(rows, 'video'), head='视频'),
           '## 5. 置信度校准', '', '| 置信度区间 | 音数 | 平均置信度 | 实际准确率 |', '|---|---|---|---|'] + cal
    if len(rel):
        md += ['', '## 6. 松键跟踪（对照 PianoVAM 真实松键时间）', '',
               f'- MIDI 结束时间被人为推后 {args.release_extra:.1f} 秒，以模拟 PM10M 转录 MIDI 的踏板延音',
               f'- 评估 {len(rel)} 个指法正确的音；误差 = 估计松键 − 真实松键',
               f'- 中位绝对误差 {np.median(np.abs(rel)) * 1000:.0f} ms；|误差|<100ms 占 {np.mean(np.abs(rel) < .1):.1%}，<200ms 占 {np.mean(np.abs(rel) < .2):.1%}',
               f'- 偏早（<−200ms）{np.mean(rel < -.2):.1%}，偏晚（>200ms）{np.mean(rel > .2):.1%}；'
               f'作为对比，直接用推后的 MIDI 结束时间误差恒为 {args.release_extra * 1000:.0f} ms',
               '- PianoVAM 没有指尖高度，这里只检验了“横向离开”判据；“抬起 15mm”判据只能在 PM10M 上靠人工复检']
    (out / 'report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps(dict(
        notes=n, coverage=len(acc_rows) / n, acc=causes['correct'] / max(len(acc_rows), 1),
        causes=dict(causes), error_causes=dict(err_causes), ece=ece,
        release_median_abs_ms=float(np.median(np.abs(rel)) * 1000) if len(rel) else None), ensure_ascii=False, indent=2))
    print('\n'.join(md))
    print('ANALYSIS_DONE ->', out, flush=True)


if __name__ == '__main__':
    main()
