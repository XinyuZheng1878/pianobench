"""Regression checks for exported timing and evidence explanations."""
import unittest
from bench_explanations import explain
from export_bench_gold import to_eval
from audit_bench import audit


class ExportTests(unittest.TestCase):
    def test_native_duration_and_explanation(self):
        row = dict(qid='test', needs_human=False, atoms=[], private_target=dict(answer='A', difficulty=dict(keys=['C#3','G3'])),
                   media=dict(frames=[dict(t=0, native_frame=1260, path='a.jpg'),
                                      dict(t=146/30, native_frame=1406, path='b.jpg')]),
                   question=dict(task='T1', subtype='和弦与音程', level='L1', video_id='v',
                                 question='选项：A. 增四度；B. 纯五度。', answer_type='choice',
                                 options=['增四度', '纯五度'], clip=dict(fps=30, fps_out=8),
                                 semantic_id='test', slots=dict(hand='左手')))
        row['atoms'] = [dict(id='k1', t=2.27, subject='左手', predicate='按下', object='C#3'),
                        dict(id='k2', t=2.27, subject='左手', predicate='按下', object='G3')]
        q = to_eval(row)
        self.assertAlmostEqual(q['clip']['length_s'], 4.9)
        self.assertIn('6 个半音', q['explanation'])
        self.assertIn('{A}', q['explanation'])
        self.assertIn('4.900 秒', q['question'])
        self.assertEqual(audit([q])['issues'], [])

    def test_forecast_does_not_invent_future_evidence(self):
        row = dict(question=dict(task='T5', subtype='下一个键（选择）', predict=dict(t0=2, horizon=.5)),
                   atoms=[], private_target=dict(answer='A'))
        result = explain(row, dict(answer='A', options=['C4', 'D4']))
        self.assertIn('未来事件明细未保存在此记录中', result)
        self.assertIn('{A}（C4）', result)

    def test_bad_choice_and_duplicate_detected(self):
        q = dict(qid='a', task='T1', subtype='test', answer_type='choice', answer='Z', options=['x','x'])
        issues = {x['issue'] for x in audit([q, q])['issues']}
        self.assertTrue({'invalid_choice_answer', 'invalid_or_duplicate_options', 'duplicate_qid'} <= issues)


if __name__ == '__main__':
    unittest.main()
