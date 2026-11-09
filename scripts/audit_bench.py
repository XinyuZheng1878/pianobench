"""Audit an exported questions.jsonl without model calls or media decoding."""
import argparse
import collections
import json
from pathlib import Path


def audit(rows):
    issues = []
    seen = set()
    by = collections.defaultdict(list)
    for q in rows:
        qid = q.get('qid')
        def flag(message):
            issues.append(dict(qid=qid, issue=message))
        if qid in seen:
            flag('duplicate_qid')
        seen.add(qid)
        by[(q.get('task'), q.get('subtype'))].append(q)
        opts = q.get('options') or []
        if q.get('answer_type') == 'choice':
            answer = q.get('answer')
            if not isinstance(answer, str) or answer not in list('ABCDEFG'[:len(opts)]):
                flag('invalid_choice_answer')
            if len(opts) < 2 or len(set(opts)) != len(opts):
                flag('invalid_or_duplicate_options')
        media = q.get('media', {})
        times = media.get('frame_times', [])
        if len(times) != len(media.get('frames', [])):
            flag('frame_time_count_mismatch')
        if any(a >= b for a, b in zip(times, times[1:])):
            flag('frame_times_not_increasing')
        clip = q.get('clip', {})
        if q.get('fps') and 'start_frame' in clip and 'end_frame' in clip:
            expected = (clip['end_frame'] - clip['start_frame']) / q['fps']
            if abs(expected - clip.get('length_s', expected)) > .001:
                flag('clip_length_mismatch')
        if q.get('predict') and times and abs(times[-1] - q['predict']['t0']) > .002:
            flag('prediction_cut_mismatch')
        if not q.get('explanation'):
            flag('missing_explanation')
    stats = []
    for (task, subtype), group in sorted(by.items()):
        letters = collections.Counter(str(q.get('answer')) for q in group if q.get('answer_type') == 'choice')
        classes = collections.Counter()
        for q in group:
            opts, a = q.get('options') or [], q.get('answer')
            label = opts[ord(a) - 65] if isinstance(a, str) and a in list('ABCDEFG'[:len(opts)]) else a
            classes[json.dumps(label, ensure_ascii=False)] += 1
        stats.append(dict(task=task, subtype=subtype, count=len(group), letters=dict(letters),
                          answer_classes=dict(classes), recordings=len({q.get('video_id') for q in group})))
    return dict(questions=len(rows), issues=issues, subtypes=stats,
                scope='结构、时间字段及分布检查；视频可辨识性、标注真实性和预测合理性需要另外复核')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--questions', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    rows = [json.loads(line) for line in Path(a.questions).read_text().splitlines() if line.strip()]
    report = audit(rows)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"{report['questions']} questions, {len(report['issues'])} findings -> {a.out}")


if __name__ == '__main__':
    main()
