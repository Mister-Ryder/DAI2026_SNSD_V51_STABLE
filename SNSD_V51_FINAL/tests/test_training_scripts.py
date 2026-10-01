"""Structure and leakage regressions for the strict 8-GPU training sweep."""

from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPOSITORY_ROOT / 'scripts'
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from summarize_training_sweep import summarize_strict_training_sweep


LAUNCHER = SCRIPT_ROOT / 'train_gnn_8gpu.sh'


def _bash_array(text, name):
    match = re.search(r"^{}=\((.*?)\)".format(re.escape(name)), text, re.M | re.S)
    if match is None:
        raise AssertionError("missing Bash array {}".format(name))
    return tuple(re.findall(r"[^\s]+", match.group(1)))


def _write_manifest(root, rows):
    fields = (
        "trial",
        "gpu",
        "init_mode",
        "learning_rate",
        "exploration_epsilon",
        "include_shuffled",
        "heads_only",
        "zero_target_semantics",
        "epochs",
        "patience",
        "trainer_seed",
        "negotiation_steps",
        "warmup_quartets",
        "train_seeds",
        "validation_seeds",
        "test_seeds",
        "pid",
        "run_dir",
        "log",
        "checkpoint",
        "initial_checkpoint",
    )
    with (root / "launch_manifest.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _step_audit(change_rate):
    return {
        "Real/step_{}".format(step): {
            "observations": 2,
            "candidate_samples": 12,
            "minimum_candidates_per_agent": 1,
            "revealed_candidates": 12,
            "anchor_change_events": int(change_rate * 12),
            "anchor_change_rate": change_rate,
            "full_agent_activity": True,
        }
        for step in range(12)
    }


class StrictTrainingSweepScriptTests(unittest.TestCase):
    def test_launcher_has_eight_controlled_cells_and_explicit_eight_runs(self):
        text = LAUNCHER.read_text(encoding="utf-8")
        trials = _bash_array(text, "TRIALS")
        self.assertEqual(8, len(trials))
        self.assertEqual(8, len(set(trials)))
        self.assertEqual(
            ("11", "22", "33", "44", "55", "66", "77", "88"),
            _bash_array(text, "RUN_DIR_SEEDS"),
        )
        self.assertIn('TRAIN_SEEDS="44,55,66,77,88"', text)
        self.assertIn('VALIDATION_SEEDS="33"', text)
        self.assertIn('TEST_SEEDS="11,22"', text)
        self.assertIn('--run-dir', text)
        self.assertIn('for warmup_seed in "${RUN_DIR_SEEDS[@]}"', text)
        self.assertIn('--negotiation-steps "$DEPTH"', text)
        self.assertIn('--device cuda:0', text)

        self.assertEqual(
            {"1e-4", "3e-4", "5e-4"}, set(_bash_array(text, "LEARNING_RATES"))
        )
        self.assertEqual({"0.05", "0.10"}, set(_bash_array(text, "EPSILONS")))
        self.assertIn("warmstart", _bash_array(text, "INIT_MODES"))
        self.assertIn("1", _bash_array(text, "INCLUDE_SHUFFLED"))
        self.assertIn("1", _bash_array(text, "HEADS_ONLY"))
        self.assertIn("global", _bash_array(text, "ZERO_SEMANTICS"))

    def test_validation_selects_before_better_test_is_attached(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for gpu in range(8):
                trial = "trial_{}".format(gpu)
                run_dir = root / trial
                run_dir.mkdir()
                # trial_0 wins validation but deliberately loses test.
                validation_loss = 1.0 if gpu == 0 else 2.0 + gpu
                test_loss = 1000.0 if gpu == 0 else 0.01 + gpu
                graph_layers = {
                    "checkpoint_hash": "hash-{}".format(gpu),
                    "best_epoch": 3,
                    "best_validation_loss": validation_loss,
                    "parameter_count": 456075,
                    "trainable_parameter_count": 456075,
                    "encoder_trainable_parameter_count": 350340,
                    "validation_metrics": {
                        "loss": validation_loss,
                        "real_rank_accuracy": 0.5,
                        "zero_rank_accuracy": 0.4,
                    },
                    "test_metrics": {
                        "loss": test_loss,
                        "real_rank_accuracy": 0.9 if gpu else 0.1,
                        "zero_rank_accuracy": 0.8 if gpu else 0.2,
                    },
                    "validation_rollout": {"per_step": _step_audit(0.5)},
                }
                (run_dir / "model_manifest_message_v4_5_strict.json").write_text(
                    json.dumps(graph_layers), encoding="utf-8"
                )
                rows.append(
                    {
                        "trial": trial,
                        "gpu": str(gpu),
                        "init_mode": "scratch",
                        "learning_rate": "3e-4",
                        "exploration_epsilon": "0.05",
                        "include_shuffled": "0",
                        "heads_only": "0",
                        "zero_target_semantics": "intrinsic",
                        "epochs": "30",
                        "patience": "6",
                        "trainer_seed": "20260803",
                        "negotiation_steps": "12",
                        "warmup_quartets": "7",
                        "train_seeds": "44,55,66,77,88",
                        "validation_seeds": "33",
                        "test_seeds": "11,22",
                        "pid": str(100 + gpu),
                        "run_dir": str(run_dir),
                        "log": str(root / "logs" / (trial + ".log")),
                        "checkpoint": str(run_dir / "checkpoint.pt"),
                        "initial_checkpoint": "",
                    }
                )
            _write_manifest(root, rows)
            result = summarize_strict_training_sweep(root, require_complete=True)
            protocol = result["selection_protocol"]
            self.assertEqual("trial_0", protocol["selected_trial"])
            self.assertFalse(protocol["test_metrics_used_for_selection"])
            self.assertEqual(
                "trial_0", result["trials_by_validation_rank"][0]["trial"]
            )
            self.assertEqual(1000.0, result["trials_by_validation_rank"][0]["postselection_test_loss"])
            self.assertEqual(96, len(result["validation_per_step"]))
            self.assertTrue(
                result["trials_by_validation_rank"][0][
                    "validation_all_steps_full_agent_activity"
                ]
            )
            self.assertAlmostEqual(
                0.5,
                result["trials_by_validation_rank"][0][
                    "validation_anchor_change_rate"
                ],
            )

    def test_partial_sweep_is_reported_and_strict_mode_rejects_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for gpu in range(8):
                run_dir = root / "missing_{}".format(gpu)
                rows.append(
                    {
                        "trial": "missing_{}".format(gpu),
                        "gpu": str(gpu),
                        "init_mode": "scratch",
                        "learning_rate": "3e-4",
                        "exploration_epsilon": "0.05",
                        "include_shuffled": "0",
                        "heads_only": "0",
                        "zero_target_semantics": "intrinsic",
                        "epochs": "30",
                        "patience": "6",
                        "trainer_seed": "20260803",
                        "negotiation_steps": "12",
                        "warmup_quartets": "7",
                        "train_seeds": "44,55,66,77,88",
                        "validation_seeds": "33",
                        "test_seeds": "11,22",
                        "pid": "0",
                        "run_dir": str(run_dir),
                        "log": str(root / "none.log"),
                        "checkpoint": str(run_dir / "checkpoint.pt"),
                        "initial_checkpoint": "",
                    }
                )
            _write_manifest(root, rows)
            result = summarize_strict_training_sweep(root)
            self.assertEqual("partial", result["status"])
            self.assertEqual(8, result["incomplete_trial_count"])
            with self.assertRaisesRegex(ValueError, "strict sweep is incomplete"):
                summarize_strict_training_sweep(root, require_complete=True)


if __name__ == "__main__":
    unittest.main()
