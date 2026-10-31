"""P0 synthetic positive/counterexample contracts; no video, network or model required."""
import copy
import random
import unittest
import numpy as np
import p0_restore as p


def perf(on=(60,120,180),off=(90,150,210),pitch=(60,64,62),finger=None,hand=None,T=600):
    n=len(on)
    return p.state.Perf("synthetic_dev",60.,T,"synthetic.mp4",np.array(pitch),np.array(on),np.array(off),
        np.array(hand if hand is not None else [1]*n),np.array(["gold"]*n),
        np.array(finger if finger is not None else [0]*n),np.full((T,2),20.),
        np.tile([18.,19.,20.,21.,23.],(T,2,1)),np.ones((T,2),bool),np.array(on)/60.)


def find(P, sub):
    for seed in range(200):
        g=p.rules.Gen(P,random.Random(seed))
        if sub in p.T3:
            f=getattr(g,p.T3[sub])(0,300)
            fs=[f] if f is not None and f["subtype"]==sub else []
        elif sub in p.T4:
            fs=[f for f in g.t4(0,360) if f["subtype"]==sub]
        elif sub in p.T5:
            fs=[f for f in g.t5(180,1) if f["subtype"]==sub]
        else:
            f=p.t2_finger(g,0,300);fs=[f] if f else []
        for f in fs:
            try:
                return p.freeze(P,f,"dev","a"*64)
            except (ValueError,IndexError):
                continue
    return None


class P0Tests(unittest.TestCase):
    def test_gold_hand_does_not_require_gold_finger(self):
        r=dict(pitch=[60,62,64],hand_pred=[0,1,1],hand_conf=[.99]*3)
        z=dict(gt_hand=[1,-1,-1],gt_finger=[-1,2,-1],label_hand=[0,1,-1])
        hand,source,finger=p.state.trust_sources(z,r)
        self.assertEqual(hand.tolist(),[1,1,-1])
        self.assertEqual(finger.tolist(),[-1,-1,-1])
        z['gt_finger']=[4,2,-1]
        self.assertEqual(p.state.trust_sources(z,r)[2].tolist(),[4,-1,-1])

    def test_grouping_is_complete_non_chained_before_filter(self):
        P=perf(on=(60,62,65),off=(100,100,100));P.on_s=np.array([1.,1.039,1.078])
        self.assertEqual(P.groups([0,1,2],.04),[[0,1],[2]])
        self.assertEqual(P.groups([1,2],.04),[[1],[2]])
        with self.assertRaises(p.GateError):p.groups_in_window(P,1,62,100)

    def test_short_press_and_repetition_require_visible_release(self):
        P=perf(on=(9,),off=(12,),pitch=(60,));g=p.rules.Gen(P,random.Random(0))
        self.assertFalse(g.press_visible(0,g.grid(0,300,8)))
        P=perf(on=(10,20),off=(25,40),pitch=(60,60));g=p.rules.Gen(P,random.Random(0))
        self.assertFalse(g.press_visible(1,g.grid(0,300,8)))
        P=perf(on=(60,120),off=(90,150),pitch=(60,60));g=p.rules.Gen(P,random.Random(0))
        self.assertTrue(g.countable([[0],[1]],g.grid(0,300,8)))
        P=perf(on=(60,),off=(60,),pitch=(60,));g=p.rules.Gen(P,random.Random(0))
        self.assertFalse(g.press_visible(0,g.grid(0,300,8)))
        with self.assertRaises(p.GateError):p.retrospective_gate(P,1,0,300,g.grid(0,300,8))

    def test_span_uses_only_thumb_little(self):
        P=perf();P.tips[60,1]=[20,14,15,16,17]
        self.assertEqual(P.span(1,60),3)
        P.palm[70:74]=np.nan
        self.assertTrue(p.rules.Gen(P,random.Random(0)).has_jump(1,60,80))

    def test_hold_bin_observation_interval(self):
        self.assertIsNone(find(perf(on=(60,),off=(63,),pitch=(60,)),"按键保持时长"))
        self.assertIsNotNone(find(perf(on=(60,),off=(150,),pitch=(60,)),"按键保持时长"))

    def test_forecast_group_cannot_be_truncated_at_horizon_or_cut(self):
        for on in [(170,209,211),(170,179,181)]:
            P=perf(on=on,off=(175,230,231),pitch=(60,62,64))
            self.assertIsNone(find(P,"下一个键（选择）"))
            self.assertIsNone(find(P,"下一根手指"))

    def test_future_key_state_includes_repress(self):
        P=perf(on=(160,200),off=(190,240),pitch=(60,60))
        row=find(P,"某个键之后是否处于按下状态")
        # Find the 0.5-second case independently (find() may return 0.2).
        f=next(f for f in p.rules.Gen(P,random.Random(0)).t5(180,1)
               if f['subtype']=='某个键之后是否处于按下状态' and f['predict']['horizon']==.5)
        self.assertEqual(f['answer'],'A')
        self.assertTrue(f['difficulty']['repressed'])
        self.assertIsNotNone(row)

    def test_repeated_previous_key_is_valid_target_and_all_options_reachable(self):
        P=perf(on=(160,200),off=(175,230),pitch=(60,60))
        row=find(P,"下一个键（选择）")
        self.assertIsNotNone(row)
        self.assertTrue(row['private_target']['difficulty']['repeat_of_last'])
        self.assertEqual(len(set(row['question']['options'])),4)
        self.assertEqual(row['question']['options'][ord(row['private_target']['answer'])-65],'C4')
        from_key={p.state.pitch_name(k):k for k in range(21,109)}
        self.assertTrue(all(abs(p.state.KEY_X[from_key[x]]-20)<=5 for x in row['question']['options']))

    def test_prediction_input_is_causal_and_truth_private(self):
        P=perf(on=(160,200),off=(175,230),pitch=(60,62))
        row=find(P,"下一个键（选择）")
        card=p.writer_input(row)
        self.assertNotIn('answer',card['question'])
        self.assertNotIn('difficulty',card['question'])
        self.assertNotIn('private_target',card)
        cut=card['question']['predict']['t0']
        self.assertEqual(cut,card['question']['frames'][-1]['t'])
        self.assertTrue(all(a['t']<=cut for a in card['atoms']))
        self.assertTrue(all(a.get('object')!='D4' for a in card['atoms'] if a['predicate']=='按下'))
        before=copy.deepcopy(card)
        P.palm[181:]=99
        # Historical raw palm atoms must remain byte-identical when future skeleton changes.
        f=next(f for f in p.rules.Gen(P,random.Random(0)).t5(180,1) if f['subtype']=='下一个键（选择）')
        self.assertEqual([a for a in before['atoms'] if a['source']=='骨架'],[a for a in f['atoms'] if a['source']=='骨架'])

    def test_finger_approval_is_hash_bound_and_analysis_relations_exact(self):
        row=find(perf(),"哪根手指")
        a=row['atoms'][0]
        clause=dict(atom_id=a['id'],text=p.atom_sentence(a))
        obs='交付画面中可见键盘，所问手部所在区域清晰可辨。'
        with self.assertRaises(p.GateError):p.compose_sft(row,obs,[clause])
        approval=dict(qid=row['qid'],fact_hash=row['fact_hash'],decision='approve',reviewer='human')
        self.assertTrue(p.compose_sft(row,obs,[clause],[approval]).endswith('所以答案是{'+row['private_target']['answer']+'}'))
        self.assertFalse(p.approved(row,[dict(approval,fact_hash='old')]))
        changed=copy.deepcopy(row);changed['private_target']['answer']='B'
        self.assertFalse(p.approved(changed,[approval]))
        with self.assertRaises(p.GateError):p.compose_sft(changed,obs,[clause],[approval])
        changed=copy.deepcopy(row);changed['needs_human']=False
        with self.assertRaises(p.GateError):p.compose_sft(changed,obs,[clause],[approval])
        with self.assertRaises(p.GateError):p.compose_sft(row,obs,[dict(clause,text='左手按下D4。')],[approval])

    def test_finger_fields_are_chinese_names_only(self):
        row=find(perf(),"哪根手指")
        self.assertEqual(row['question']['options'],p.state.FINGER_ZH)
        self.assertTrue(row['needs_human'])
        self.assertIsNone(find(perf(finger=[-1,-1,-1]),"哪根手指"))

    def test_first_melody_peak_near_boundary_not_replaced_by_second(self):
        P=perf(on=(6,18,30,120,150,180),off=(12,25,40,140,170,210),pitch=(60,64,62,60,65,62))
        self.assertIsNone(find(P,"旋律最高点"))

    def test_each_restored_subtype_positive_and_missing_observation_counterexample(self):
        base=perf()
        move=perf();move.palm[100:141]=np.linspace(20,28,41)[:,None];move.palm[141:]=28
        pairs=perf(on=(60,61),off=(120,150),pitch=(60,64))
        dual=perf(on=(60,120),off=(180,210),pitch=(60,64),hand=(0,1))
        held=perf(on=(60,),off=(150,),pitch=(60,))
        chord=perf(on=(60,60,60,180),off=(100,100,100,210),pitch=(60,64,67,72))
        future=perf(on=(160,200),off=(190,240),pitch=(60,62))
        cases={'哪根手指':base,'按键次数·整只手':base,'按键次数·指定手指':base,
               '按键保持时长':held,'张开跨度':base,'手掌净位移':move,'持键重叠':perf(on=(60,120),off=(180,210),pitch=(60,64)),'双手配合':dual,
               '先松开哪个键':pairs,'旋律最高点':base,'第一次三音以上同时按下':chord,
               '第一次按进新的八度':chord,'第一次大幅横移':move,'三音键序列':base,
               '指定手指按指定键':base,'长按某个键':held,
               **{s:future for s in p.T5}}
        for subtype,P in cases.items():
            with self.subTest(subtype=subtype,case='positive'):
                row=find(P,subtype)
                self.assertIsNotNone(row,subtype)
                if row is not None:
                    self.assertEqual(row['question']['subtype'],subtype)
            bad=copy.deepcopy(P);bad.ok[:]=False;bad.palm[:]=np.nan;bad.tips[:]=np.nan
            with self.subTest(subtype=subtype,case='negative'):
                self.assertIsNone(find(bad,subtype),subtype)


class P0APIContractTests(unittest.TestCase):
    def call_with_fake(self, row, approvals=(), hard_error=False):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import p0_api
        class Fake:
            def __init__(self):self.calls=[]
            def call(self, model, system, content, max_tokens=1400):
                self.calls.append((system,content))
                trace=dict(model='offline-fake',raw='{}',usage={},seconds=0,retries=0)
                if system==p0_api.B_PROMPT:return dict(visible=True,keyboard_visible=True,target_region_visible=True),trace
                if system==p0_api.D_PROMPT:return dict(atom_ids=[row['atoms'][0]['id']],style='concise'),trace
                return dict(hard_error=hard_error,faithful=9,clear=9,comment='offline fixture'),trace
        api=Fake()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder, patch('p0_api.delivery_content',return_value=[]):
            out=Path(folder)
            result=p0_api.produce_one(api,out,out,row,approvals)
            if result['status']=='auto_accepted':
                again=p0_api.produce_one(api,out,out,row,approvals)
                self.assertEqual(again['answer'],result['answer'])
                self.assertEqual(len(api.calls),3)
        return result,api.calls

    def test_prediction_API_inputs_do_not_include_future_truth(self):
        row=find(perf(on=(160,200),off=(175,230),pitch=(60,62)),"下一个键（选择）")
        result,calls=self.call_with_fake(row)
        self.assertEqual(result['status'],'auto_accepted')
        self.assertEqual(len(calls),3)
        import json
        d=json.loads(calls[1][1])
        self.assertNotIn('private_target',d)
        self.assertNotIn('answer',d['question'])
        self.assertTrue(all(a['t']<=d['question']['predict']['t0'] for a in d['atoms']))
        self.assertIn('private_target',json.loads(calls[2][1]))
        self.assertTrue(result['answer'].endswith('所以答案是{'+row['private_target']['answer']+'}'))

    def test_finger_requires_real_hash_bound_approval_before_calls(self):
        row=find(perf(),"哪根手指")
        result,calls=self.call_with_fake(row)
        self.assertEqual(result['status'],'needs_human_GT_review')
        self.assertEqual(calls,[])
        approval=dict(qid=row['qid'],fact_hash=row['fact_hash'],decision='approve',reviewer='human')
        result,calls=self.call_with_fake(row,[approval])
        self.assertEqual(result['status'],'human_accepted')
        self.assertTrue(result['human_verified'])
        self.assertEqual(len(calls),3)

    def test_judge_hard_error_cannot_enter_accepted_export(self):
        row=find(perf(),"按键次数·整只手")
        result,calls=self.call_with_fake(row,hard_error=True)
        self.assertEqual(result['status'],'needs_review')
        self.assertNotIn('answer',result)
        import tempfile,json
        from pathlib import Path
        import p0_api
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            out=Path(folder).resolve();source=out.parent/'frozen_source'
            with self.assertRaises(ValueError):p0_api.guard_output(source,out)
            self.assertEqual(list(out.iterdir()),[])
            (out/'p0_api_manifest.json').write_text(json.dumps(dict(profile='p0-api-dev-v1',source=str(source))),encoding='utf-8')
            p0_api.guard_output(source,out)


if __name__=='__main__':unittest.main(verbosity=2)
