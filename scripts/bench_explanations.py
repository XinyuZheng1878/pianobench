"""Evidence-based offline explanations for gold benchmark exports."""
import collections
import re


def pitch_number(name):
    match = re.fullmatch(r'([A-G])(#?)(-?\d+)', name)
    if not match:
        raise ValueError(f'Invalid pitch {name}')
    note, sharp, octave = match.groups()
    return 12 * (int(octave) + 1) + dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)[note] + bool(sharp)


def explain(row, exported):
    q = row['question']
    slots = q.get('slots', {})
    span = q.get('range') or {}
    start, end = span.get('start', 0), span.get('end', float('inf'))
    atoms = [a for a in row.get('atoms', []) if start <= a['t'] < end
             and (not slots.get('hand') or a['subject'] == slots['hand'])]
    keys = {slots[k] for k in ('key', 'key_a', 'key_b') if k in slots}
    if keys:
        atoms = [a for a in atoms if a.get('object') in keys]
    if slots.get('finger'):
        atoms = [a for a in atoms if a.get('finger') == slots['finger']]
    events = [a for a in atoms if a['predicate'] == '按下']
    answer = exported['answer']
    label = exported['options'][ord(answer) - ord('A')] if exported.get('options') else answer
    if isinstance(label, list):
        label = '、'.join(map(str, label))
    if q["task"] == "T3" and q["subtype"] == "手掌净位移":
        endpoints = {a["id"]: a for a in row.get("atoms", [])}
        if {"g_start", "g_end"} <= endpoints.keys():
            first, last = (endpoints[key] for key in ("g_start", "g_end"))
            times = exported["media"]["frame_times"]
            observations = []
            for point in (first, last):
                index = min(range(len(times)), key=lambda i: abs(times[i] - point["t"]))
                observations.append(
                    f'第 {index} 帧（{times[index]:.3f} 秒），'
                    f'{point["subject"]}掌心在 {point["object"]} 附近'
                )
            delta = last["x"] - first["x"]
            direction = "向左" if delta < 0 else "向右"
            movement = "净位移不足 1 个白键宽" if abs(delta) < 1 else f"{direction} {abs(delta):.2f} 个白键宽"
            return (
                "【观察】" + "。".join(observations) + "。"
                f'【分析】用端点附近的掌心位置估计，终点减起点为 '
                f'{last["x"]:.2f} − {first["x"]:.2f} = {delta:.2f} 个白键宽，'
                f'即{movement}，对应“{label}”。所以答案是{{{answer}}}。'
            )
    evidence = []
    reasoning = []
    if q['task'] == 'T5':
        cut = q['predict']['t0']
        reasoning.append(f'预测截止点为 {cut:.3f} 秒，预测跨度为 {q["predict"]["horizon"]:g} 秒。'
                         '答案由原始录制中截止点之后的标注确定，历史画面并不保证未来结果唯一。')
        evidence.append('当前记录的 atoms 仅含截止点及以前的证据，未来事件明细未保存在此记录中。')
    else:
        evidence = [f'{a["t"]:.3f} 秒，{a["subject"]}{a.get("finger", "")}{a["predicate"]}{a["object"]}'
                    f'（证据 {a["id"]}）' for a in atoms if a['predicate'] in ('按下', '松开')]
        sub = q['subtype']
        diff = row['private_target'].get('difficulty', {})
        if sub == '和弦与音程':
            notes = sorted(diff.get('keys', []), key=pitch_number)
            if notes:
                reasoning.append('生成器按连续 MIDI 起音时间分组，目标组包含 ' + '、'.join(notes) + '。')
                if len(notes) == 2:
                    reasoning.append(f'两键相距 {pitch_number(notes[1]) - pitch_number(notes[0])} 个半音，'
                                     '按本数据集的半音数音程命名约定判定。')
        elif sub.startswith('按键次数'):
            reasoning.append(f"筛选指定手或手指后，按首个起音后 40 毫秒分组，生成器记录为 {row['private_target']['answer']} 次；"
                             '整只手计数题中，一组算一次。')
        elif sub == '按键序列':
            reasoning.append('按起音时间排序并保留重复键，得到 ' + ' → '.join(a['object'] for a in events) + '。')
        elif sub == '音区' and events:
            counts = collections.Counter(pitch_number(a['object']) // 12 - 1 for a in events)
            octave, count = counts.most_common(1)[0]
            reasoning.append(f'C{octave}–B{octave} 范围占 {count}/{len(events)} 次按下，比例为 {count / len(events):.1%}。')
        elif sub == '织体':
            sizes = diff.get('sizes', [])
            reasoning.append(f'按 40 毫秒起音组统计，各组不同键数为 {sizes}。')
        diff = row['private_target'].get('difficulty', {})
        for key, text, unit in [('hold_s', '按键保持时长', '秒'), ('span', '手部跨度', '个白键宽'),
                                ('dx', '有符号横向位移', '个白键宽'), ('release_gap_s', '两键松开时间差', '秒'),
                                ('gap_s', '前键松开时刻减去后键按下时刻', '秒')]:
            if key in diff:
                reasoning.append(f'{text}为 {diff[key]} {unit}。')
        if q['answer_type'] in ('time_point', 'time_span'):
            reasoning.append(f'标注定位结果为 {row["private_target"]["answer"]} 秒，按题目要求匹配时间选项。')
    if row.get('needs_human'):
        reasoning.append('指法依据数据标签；上音哈农使用谱面推导指法，实际演奏指法需视频复核。')
    evidence_text = '；'.join(evidence) if evidence else '此题依据派生标注，当前筛选范围无可列出的按键事件。'
    conclusion = f'答案为 {{{answer}}}' + (f'（{label}）' if exported.get('options') else '') + '。'
    return '【标注依据】' + evidence_text + '。【判定】' + ''.join(reasoning) + '【结论】' + conclusion
