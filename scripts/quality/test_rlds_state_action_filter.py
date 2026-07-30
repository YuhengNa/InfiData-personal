from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/quality/filter_rlds_state_action.py"
SPEC = importlib.util.spec_from_file_location("state_action_filter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class StateActionFilterTests(unittest.TestCase):
    def test_s1_detects_injected_spike(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 160))[:, None]
        x[80, 0] += 20
        result = MODULE.detect_s1(x, MODULE.S1Config())
        self.assertTrue(result["flagged"])
        self.assertTrue(any(abs(frame - 80) <= 1 for frame in result["frames"]))

    def test_s1_can_ignore_discrete_gripper_dimension(self):
        x = np.zeros((160, 2))
        x[80:, 1] = 1
        result = MODULE.detect_s1(x, MODULE.S1Config(), ignored_dimensions={1})
        self.assertFalse(result["flagged"])

    def test_s2_finds_action_lead(self):
        action = np.sin(np.linspace(0, 10 * np.pi, 240))[:, None]
        state = np.concatenate([np.zeros((3, 1)), action[:-3]], axis=0)
        result = MODULE.detect_s2(
            state, action, MODULE.S2Config(max_lag=8, min_active=20),
        )
        dim = result["dimensions"][0]
        self.assertEqual(dim["lag"], 3)
        self.assertGreater(dim["directional_agreement"], 0.9)
        self.assertFalse(result["flagged"])

    def test_s2_flags_opposite_trend(self):
        action = np.sin(np.linspace(0, 10 * np.pi, 240))[:, None]
        state = -action
        result = MODULE.detect_s2(
            state, action, MODULE.S2Config(max_lag=0, min_active=20),
        )
        self.assertTrue(result["flagged"])
        self.assertLess(result["dimensions"][0]["directional_agreement"], 0.1)

    def test_s2_rejects_mismatched_layout(self):
        metadata = {
            "state_action_schema_json": {
                "state_layout": ["joint_1", "joint_2"],
                "action_layout": ["eef_x", "eef_y"],
            }
        }
        compatible, reason = MODULE.s2_is_compatible(metadata, 2, 2)
        self.assertFalse(compatible)
        self.assertEqual(reason, "state_action_layout_mismatch")

    def test_s3_detects_extreme_and_ignores_gripper(self):
        rng = np.random.default_rng(3)
        normal = rng.normal(0, 1, size=(500, 2))
        episode = MODULE.Episode("fit", normal, normal.copy(), {})
        thresholds = MODULE.fit_s3(
            [episode], MODULE.S3Config(alpha=1.5), {1}, {1},
        )
        probe = np.array([[100.0, 100.0]])
        result = MODULE.detect_s3(probe, thresholds["state"], {1})
        self.assertEqual(result["frames"], [0])
        self.assertEqual([hit["dim"] for hit in result["hits"]], [0])


if __name__ == "__main__":
    unittest.main()
