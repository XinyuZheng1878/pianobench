"""Opt-in API adapter for frozen P0 DEV candidates; importing/running tests makes no requests."""
import argparse
import base64
import json
from pathlib import Path
import sys
from batch_sft import API, ModelReplyError, atomic_json, digest, file_hash, judge_pass, read_json
from p0_restore import (ROOT, FINGER, approved, atom_sentence, compose_sft, verify_frozen, writer_input)

CONVENTIONS = "左/右手指演奏者本人；画面左为低音。时间以第一张交付帧为0，最后一张预测帧就是现在。40ms组以最早按下锚定，不链式延长。手指仅指人工标注的按下瞬间手指，不能推断持键中途用指、五态、穿指跨指或踏板。"
B_PROMPT = "【任务】仅检查交付帧是否清楚可供复核，不回答题目。"+CONVENTIONS+"【输出】严格只输出JSON，三个字段必须是布尔值：visible（所问时段画面清晰连续）、keyboard_visible（目标琴键区域可见）、target_region_visible（目标手部可辨）。示例：{\"visible\":true,\"keyboard_visible\":true,\"target_region_visible\":true}。不得添加动作、时刻、答案或未来叙述。"
D_PROMPT = "【任务】从固定问题和过去的原子中选择证据，不写答案。"+CONVENTIONS+"【输入】question含固定题干和选项；atoms中id是唯一引用标识。未来目标不提供。每条原子是不可拆分的手/指/动作/键/时刻关系。【输出】只输出JSON，atom_ids是最多12个不同的有效id；style枚举chronological或concise。示例：{\"atom_ids\":[\"k1\"],\"style\":\"chronological\"}。不要增加自由文本或答案字段。"
J_PROMPT = "【任务】检查SFT文字与事实关系。"+CONVENTIONS+"【输入】private_target仅供核对标准答案，不是已见画面；atoms仅为历史/交付区间事实。【输出】仅JSON：{\"hard_error\":false,\"faithful\":9,\"clear\":9,\"comment\":\"简短理由\"}。hard_error严格布尔，faithful和clear严格0至10整数。手/指/键/时间/动作绑定错误、最终答案错误、预测未来冒充观察，均hard_error=true一票否决。你只检查文字，不宣称亲眼核对视频。"
OBSERVATION = "交付画面中可见键盘，所问手部所在区域清晰可辨。"


def delivery_content(source, row):
    import cv2
    import numpy as np
    frames = row['question']['frames']
    actual = row['media']['frames']
    if ([f['video_frame'] for f in frames] != [f['native_frame'] for f in actual]
            or any(abs(f['t']-m['t'])>1e-9 for f,m in zip(frames,actual))):
        raise ValueError('question/delivery timestamp mismatch')
    content=[dict(type='text',text=row['question']['question'])]
    for item in row['media']['packs']:
        path=source/item['path']
        if not path.resolve().is_relative_to(source.resolve()) or file_hash(path)!=item['sha256']:
            raise ValueError('delivery pack missing/changed')
        data=path.read_bytes()
        if cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR) is None:
            raise ValueError('delivery JPEG not decodable')
        content.append(dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(data).decode())))
    return content


def select_analysis(value, row):
    if set(value)!={'atom_ids','style'} or value['style'] not in ('chronological','concise'):
        raise ValueError('D must select IDs/style only')
    by_id={a['id']:a for a in row['atoms']};ids=value['atom_ids']
    if (not isinstance(ids,list) or not 1<=len(ids)<=12 or any(not isinstance(i,str) or i not in by_id for i in ids)
            or len(set(ids))!=len(ids)):
        raise ValueError('invalid evidence IDs')
    # Geometry endpoints and all answer-relevant onset relations cannot be silently omitted.
    q=row['question'];sub=q['subtype'];hand=q['slots'].get('hand');interval=q.get('range')
    mandatory=[a['id'] for a in row['atoms'] if a['id'].startswith('g_')]
    if q['task']!='T5':
        for a in row['atoms']:
            in_range=(not interval or interval['type']!='timestamp' or interval['start']<=a['t']<interval['end'])
            if a['predicate']=='按下' and (hand is None or a['subject']==hand) and in_range:mandatory.append(a['id'])
    ids=set(ids)|set(mandatory)
    return [dict(atom_id=a['id'],text=atom_sentence(a)) for a in sorted(row['atoms'],key=lambda a:a['t']) if a['id'] in ids]


def produce_one(api, source, out, row, approvals=()):
    verify_frozen(row)
    if row['question']['split']!='dev':raise ValueError('P0 API pilot accepts DEV only')
    key=digest([row,[a for a in approvals if a.get("qid")==row["qid"]],B_PROMPT,D_PROMPT,J_PROMPT,
                {str(p.relative_to(ROOT)):file_hash(p) for p in ROOT.rglob('*.py') if '.git' not in p.parts}])
    dest=out/'results'/(row['qid']+'.json')
    if dest.exists():
        old=read_json(dest)
        if old.get('input_sha256')==key:
            if old.get('status') in ('auto_accepted','human_accepted'):delivery_content(source,row)
            return old
        if not (old.get('status')=='needs_human_GT_review' and old.get('fact_hash')==row['fact_hash'] and approved(row,approvals)):
            raise ValueError('cached P0 request changed')
    record=dict(row,api=[],input_sha256=key,status='needs_review',human_approvals=[a for a in approvals if a.get('qid')==row['qid']])
    # Human must approve the exact onset-finger fact before even spending model calls.
    if row['needs_human'] and not approved(row,approvals):
        record['status']='needs_human_GT_review'
        atomic_json(dest,record)
        return record
    stage='delivery'
    try:
        content=delivery_content(source,row)
        stage='B';obs,trace=api.call('Qwen3.8-27B',B_PROMPT,content,700)
        record['api'].append(dict(stage=stage,**trace))
        fields=('visible','keyboard_visible','target_region_visible')
        if set(obs)!=set(fields) or any(type(obs[k]) is not bool or not obs[k] for k in fields):
            raise ValueError('B visibility uncertain/malformed')
        stage='D';value,trace=api.call('Qwen3.8-27B',D_PROMPT,json.dumps(writer_input(row),ensure_ascii=False))
        record['api'].append(dict(stage=stage,**trace))
        clauses=select_analysis(value,row)
        answer=compose_sft(row,OBSERVATION,clauses,approvals)
        stage='judge';j,trace=api.call('dsv4-pro-0813',J_PROMPT,json.dumps(dict(question=row['question'],atoms=row['atoms'],private_target=row['private_target'],sft_answer=answer),ensure_ascii=False),700)
        record['api'].append(dict(stage=stage,**trace))
        if not judge_pass(j):raise ValueError('judge hard error/low/malformed scores')
        record.update(status='human_accepted' if row['needs_human'] else 'auto_accepted',
                      human_verified=row['needs_human'],answer=answer,analysis=clauses,judge=j)
    except Exception as e:
        if isinstance(e,ModelReplyError):record['api'].append(dict(stage=stage,**e.trace))
        record['error']=type(e).__name__+': '+str(e)[:200]
    atomic_json(dest,record)
    return record



def guard_output(source, out):
    if out==source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError('separate API output required')
    marker=out/'p0_api_manifest.json'
    if out.exists() and (not marker.is_file() or read_json(marker)!=dict(profile='p0-api-dev-v1',source=str(source))):
        raise ValueError('refuse existing non-P0 directory; production must remain read-only')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--api-doc',required=True);ap.add_argument('--approvals')
    a=ap.parse_args();source=Path(a.source).resolve();out=Path(a.out).resolve()
    guard_output(source,out)
    protected=Path('/inspire/hdd/global_user/xuguangjun-253208580299')
    if str(out).startswith(str(protected)) and not out.is_relative_to(protected/'users/qihengwang'):raise SystemExit('own directory only')
    rows=[read_json(p) for p in sorted((source/'results').glob('*.json'))]
    if not rows or any(r['question']['split']!='dev' or r['status']=='delivery_failed' for r in rows):raise SystemExit('frozen DEV candidates required')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'p0_api_manifest.json',dict(profile='p0-api-dev-v1',source=str(source)))
    import fcntl
    lock=open(out/'api.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    approvals=read_json(a.approvals) if a.approvals else []
    if not isinstance(approvals,list):raise SystemExit('approvals must be JSON list')
    media=out/'media'
    if not media.exists():media.symlink_to(source/'media',target_is_directory=True)
    elif media.resolve()!=(source/'media').resolve():raise SystemExit('unexpected media destination')
    api=API(a.api_doc)
    exports=[]
    for row in rows:
        result=produce_one(api,source,out,row,approvals)
        if result['status'] in ('auto_accepted','human_accepted'):
            exports.append(dict(id=row['qid'],video_id=row['question']['video_id'],semantic_id=row['question']['semantic_id'],
                 fact_hash=row['fact_hash'],human_verified=result['human_verified'],images=[m['path'] for m in row['media']['packs']],
                 messages=[dict(role='system',content=CONVENTIONS),dict(role='user',content=row['question']['question']),dict(role='assistant',content=result['answer'])]))
        if any(x in result.get('error','') for x in ('HTTP 401','HTTP 403')):raise SystemExit('API authorization failed; stop')
    (out/'sft_dev.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in exports),encoding='utf-8')
    atomic_json(out/'status.json',dict(processed=len(rows),accepted=len(exports)))


if __name__=='__main__':main()
