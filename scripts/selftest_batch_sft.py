"""Offline regression tests for production-only isolation and publication contracts."""
import unittest
import ast
from pathlib import Path
import tempfile
import numpy as np
import csv
from types import SimpleNamespace
from batch_sft import (answer_text, final_answer, grid_offsets, judge_pass,
                       partition, validate_analysis, shuffle_choices, check_event_visibility, decode_reply, ModelReplyError, controlled_analysis, program_summary)


class ProductionTests(unittest.TestCase):
    def test_free_visual_narrative_cannot_enter_controlled_analysis(self):
        with self.assertRaises(ValueError):
            controlled_analysis(dict(atom_ids=[], style="concise", text="2.25秒起未再按键。"), [], dict(task="T3"))

    def test_counter_uses_all_atoms_and_checks_program_truth(self):
        q = dict(task="T3", subtype="按键次数（手）", answer_type="choice", answer="A", options=["2", "3"])
        atoms = [dict(atom_id="e1", kind="key_press", hand="左手", key="D4", press_s=2.0),
                 dict(atom_id="e2", kind="key_press", hand="左手", key="E4", press_s=2.7)]
        atoms.append(program_summary(q, atoms))
        steps = controlled_analysis(dict(atom_ids=["e1"], style="chronological"), atoms, q)
        self.assertTrue(any("2.7秒" in x["text"] for x in steps))
        self.assertTrue(any("共有2次" in x["text"] for x in steps))
        with self.assertRaises(ValueError):
            program_summary(dict(q, answer="B"), atoms[:2])

    def test_controlled_analysis_rejects_invented_atom(self):
        with self.assertRaises(ValueError):
            controlled_analysis(dict(atom_ids=["invented"], style="concise"), [], dict(task="T1"))

    def test_forecast_summary_contains_no_future_event(self):
        q = dict(task="T5", subtype="会不会按键", answer_type="choice", answer="B", options=["会", "不会"],
                 predict=dict(cut=2.0, horizon=0.2))
        summary = program_summary(q, [])
        self.assertIn("作短时预测", summary["text"])
        self.assertNotIn("没有", summary["text"])

    def test_invalid_json_preserves_real_usage_and_raw_reply(self):
        result = dict(choices=[dict(message=dict(content="{'bad':true}"), finish_reason="stop")], usage=dict(total_tokens=123))
        with self.assertRaises(ModelReplyError) as caught:
            decode_reply(result, "test-model", 0, 0)
        self.assertEqual(caught.exception.trace["usage"]["total_tokens"], 123)
        self.assertEqual(caught.exception.trace["raw"], "{'bad':true}")

    def test_40ms_boundary_rejects_quantization_ambiguity(self):
        with self.assertRaises(ValueError):
            check_event_visibility([0, 2], [10, 12], [60, 64], [0.0, 0.041], [[0], [1]], [0, 8], 60, True)

    def test_repeated_key_needs_visible_release(self):
        with self.assertRaises(ValueError):
            check_event_visibility([0, 15], [12, 25], [60, 60], [0.0, 0.25], [[0], [1]], [0, 8, 15, 23], 60, False)

    def test_clear_repeated_key_and_group_boundary_pass(self):
        check_event_visibility([0, 15], [7, 25], [60, 60], [0.0, 0.25], [[0], [1]], [0, 8, 15, 23], 60, True)

    def test_shuffle_keeps_answer_content_and_copy_baseline(self):
        q = dict(answer_type="choice", options=["会", "不会"], answer="A",
                 question="会不会按键？\nA. 会\nB. 不会\n把字母写在花括号里。", baselines=dict(copy="B"))
        rng = SimpleNamespace(permutation=lambda n: [1, 0])
        shuffle_choices(q, rng)
        self.assertEqual(q["answer"], "B")
        self.assertEqual(q["options"][ord(q["answer"]) - 65], "会")
        self.assertEqual(q["baselines"]["copy"], "A")
        self.assertIn("A. 不会\nB. 会", q["question"])

    def test_video_metadata_fallback_without_ffprobe(self):
        tree = ast.parse((Path(__file__).parent / "export_pianovam.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "video_info")
        values = {1: 1920, 2: 1080, 3: 60, 4: 1200}
        cap = SimpleNamespace(isOpened=lambda: True, get=lambda k: values[k], release=lambda: None)
        cv = SimpleNamespace(VideoCapture=lambda p: cap, CAP_PROP_FRAME_WIDTH=1,
                             CAP_PROP_FRAME_HEIGHT=2, CAP_PROP_FPS=3, CAP_PROP_FRAME_COUNT=4)
        env = {"cv2": cv, "np": np, "shutil": SimpleNamespace(which=lambda p: None)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "video_info", "exec"), env)
        self.assertEqual(env["video_info"]("video.mp4"), (1920, 1080, 60.0, 1200))

    def test_tsv_reader_and_unknown_fingers_without_pandas(self):
        tree = ast.parse((Path(__file__).parent / "export_pianovam.py").read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("read_tsv", "labels")]
        env = {"np": np, "csv": csv}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "tsv_readers", "exec"), env)
        for prefix in ("", "#"):
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "sample.tsv"
                p.write_text(prefix + "onset\tkey_offset\tframe_offset\tnote\tvelocity\thand\tfinger\n"
                             "1.25\t1.5\t2.0\t60\t80\tL\t2\n"
                             "2.0\t2.2\t2.8\t64\t90\tR\t?\n", encoding="utf-8")
                data = env["read_tsv"](p)
                h, f = env["labels"](data)
                self.assertEqual(data["note"].tolist(), [60, 64])
                self.assertEqual(data["key_offset"].tolist(), [1.5, 2.2])
                self.assertEqual(h.tolist(), [0, -1])
                self.assertEqual(f.tolist(), [1, -1])

    def test_gold_and_related_piece_never_train(self):
        entries = [dict(record_time="gold", composer="x", piece="work", duplicate_group="same-performance"),
                   dict(record_time="related", composer="x", piece="work", duplicate_group="same-performance"),
                   dict(record_time="other", composer="y", piece="else")]
        p = partition(entries, {"gold"}, 0)
        self.assertEqual(p, {"gold": "final_test", "related": "heldout_related", "other": "train"})

    def test_development_piece_group_stays_together(self):
        entries = [dict(record_time=x, composer="x", piece="work", duplicate_group="same-performance") for x in ("a", "b")]
        self.assertEqual(set(partition(entries, set(), 1).values()), {"dev"})

    def test_prediction_last_frame_is_now(self):
        frames = grid_offsets(121, 60, 15, True)
        self.assertEqual(frames[-1], 120)
        self.assertEqual(frames[-1] / 60, 2.0)

    def test_nonintegral_prediction_grid_includes_last(self):
        self.assertEqual(grid_offsets(122, 60, 15, True)[-1], 121)

    def test_regular_grid_is_half_open(self):
        self.assertLess(grid_offsets(120, 60, 8)[-1], 120)

    def test_answer_is_program_letter(self):
        q = dict(answer="B", answer_type="choice")
        text = final_answer(["右手连续按键。"], [dict(text="目标区间有连续按下动作。")], q)
        self.assertEqual(text, "【观察】右手连续按键。【分析】目标区间有连续按下动作。所以答案是{B}")

    def test_model_cannot_supply_final_answer(self):
        with self.assertRaises(ValueError):
            final_answer(["所以答案是{A}"], [dict(text="有按键动作。")], dict(answer="B"))

    def test_judge_hard_error_veto(self):
        self.assertFalse(judge_pass(dict(hard_error=True, faithful=10, clear=10)))

    def test_judge_scores_are_bounded_integers(self):
        for score in (11, "9", 9.5, True):
            self.assertFalse(judge_pass(dict(hard_error=False, faithful=score, clear=10)))

    def test_bad_atom_key_hand_or_time_rejected(self):
        atoms = [dict(atom_id="e1", key="E4", hand="右手", press_s=1.2)]
        for text in ("右手在约1.2秒按下G4。", "左手在约1.2秒按下E4。", "右手在约2.2秒按下E4。"):
            with self.assertRaises(ValueError):
                validate_analysis(dict(analysis=[dict(atom_ids=["e1"], text=text)]), atoms, dict(task="T1"))

    def test_missing_atom_and_finger_claim_rejected(self):
        atoms = [dict(atom_id="e1", key="E4", hand="右手", press_s=1.2)]
        for ids, text in ((["invented"], "右手按下E4。"), (["e1"], "右手用拇指按下E4。")):
            with self.assertRaises(ValueError):
                validate_analysis(dict(analysis=[dict(atom_ids=ids, text=text)]), atoms, dict(task="T1"))

    def test_supported_clause_passes(self):
        atoms = [dict(atom_id="e1", key="E4", hand="右手", press_s=1.2)]
        value = dict(analysis=[dict(atom_ids=["e1"], text="右手在约1.2秒按下E4。")])
        self.assertEqual(validate_analysis(value, atoms, dict(task="T1")), value["analysis"])

    def test_prediction_future_evidence_rejected(self):
        atoms = [dict(atom_id="visual_context", kind="visual_context")]
        with self.assertRaises(ValueError):
            validate_analysis(dict(analysis=[dict(atom_ids=["visual_context"], text="切点之后没有按下任何键。")]),
                              atoms, dict(task="T5"))


if __name__ == "__main__":
    unittest.main()
