from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("filter_rlds_state_action.py")
SPEC = importlib.util.spec_from_file_location("state_action_filter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class StateActionFilterTests(unittest.TestCase):
    def test_episode_key_prefers_explicit_global_key(self):
        metadata = {
            "global_episode_key": "RoboMIND_shard_1:4164",
            "episode_index": 4164,
            "source_episode_index": 98,
        }
        self.assertEqual(MODULE._episode_key(metadata, 7), "RoboMIND_shard_1:4164")

    def test_episode_key_uses_episode_index_without_global_key(self):
        metadata = {"episode_index": 4164, "source_episode_index": 98}
        self.assertEqual(MODULE._episode_key(metadata, 7), "4164")

    def test_episode_key_falls_back_to_ordinal(self):
        self.assertEqual(MODULE._episode_key({"source_episode_index": 98}, 7), "7")

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

    def test_s1_uses_upper_tail_thresholds_only(self):
        x = np.zeros((20, 1))
        thresholds = {
            name: {"center": [0.1], "scale": [0.005], "threshold": [0.14]}
            for name in ("residual", "acceleration", "jerk")
        }
        result = MODULE.detect_s1(x, MODULE.S1Config(), thresholds=thresholds)
        self.assertFalse(result["flagged"])

    def test_s1_excludes_finite_difference_boundaries(self):
        x = np.arange(9.0)[:, None]
        _, metrics = MODULE._s1_metrics(x, MODULE.S1Config())

        acceleration = metrics["acceleration"][:, 0]
        self.assertTrue(np.isnan(acceleration[[0, -1]]).all())
        np.testing.assert_allclose(acceleration[1:-1], 0.0)

        jerk = metrics["jerk"][:, 0]
        self.assertTrue(np.isnan(jerk[[0, 1, -2, -1]]).all())
        np.testing.assert_allclose(jerk[2:-2], 0.0)

    def test_s1_preserves_nonfinite_gripper_flags(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                x = np.zeros((160, 2))
                x[80:, 1] = 1
                x[40, 1] = value
                result = MODULE.detect_s1(x, MODULE.S1Config(), ignored_dimensions={1})
                self.assertTrue(result["flagged"])
                self.assertEqual(result["frames"], [40])
                self.assertEqual([hit["dim"] for hit in result["hits"]], [1])

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

    def test_s2_treats_sub_epsilon_state_motion_as_stationary(self):
        delta = 0.1 * np.sin(np.linspace(0, 8 * np.pi, 240))
        action = np.cumsum(delta)[:, None]
        state = (np.cumsum(delta) * 1e-7)[:, None]
        result = MODULE.detect_s2(
            state,
            action,
            MODULE.S2Config(
                max_lag=0,
                min_active=20,
                motion_epsilon=1e-5,
                flag_negative_lag=False,
            ),
        )
        dim = result["dimensions"][0]
        self.assertEqual(dim["directional_agreement"], 0.0)
        self.assertTrue(dim["bad_da"])
        self.assertTrue(result["flagged"])

    def test_s2_ignores_gripper_dimension(self):
        length = 300
        joint_action = np.sin(np.linspace(0, 12 * np.pi, length))
        joint_state = np.concatenate([np.zeros(2), joint_action[:-2]])
        gripper_state = ((np.arange(length) // 12) % 2).astype(float)
        gripper_action = np.concatenate([np.zeros(3), gripper_state[:-3]])
        state = np.column_stack([joint_state, gripper_state])
        action = np.column_stack([joint_action, gripper_action])

        result = MODULE.detect_s2(
            state,
            action,
            MODULE.S2Config(max_lag=8, min_active=10),
            ignored_dimensions={1},
        )

        self.assertFalse(result["flagged"])
        self.assertEqual(result["ignored_dimensions"], [1])
        self.assertEqual(result["dimensions"][0]["lag"], 2)
        self.assertEqual(
            result["dimensions"][1],
            {"dim": 1, "evaluated": False, "reason": "ignored_dimension"},
        )

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

    def test_s2_parses_action_is_delta_strictly(self):
        cases = (
            (True, True),
            (False, False),
            (np.bool_(True), True),
            (np.bool_(False), False),
            ("true", True),
            (" TRUE ", True),
            ("false", False),
            (" False ", False),
            ("unknown", None),
            ("", None),
            (None, None),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertIs(MODULE.parse_action_is_delta(raw), expected)

    def test_s2_defaults_unknown_action_semantics_to_absolute(self):
        cases = (
            ("unknown", (False, "default_absolute")),
            ("", (False, "default_absolute")),
            (None, (False, "default_absolute")),
            ("false", (False, "metadata")),
            ("true", (True, "metadata")),
        )
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(MODULE.resolve_action_is_delta(raw), expected)

    def test_gripper_resolution_expands_grouped_layout(self):
        metadata = {
            "state_action_schema_json": {
                "state_dim": 8,
                "state_layout": ["franka_joint_position[7]", "gripper_position[1]"],
            }
        }
        indices, resolution = MODULE.resolve_gripper_indices(
            "auto", metadata, "state", 8,
        )
        self.assertEqual(indices, {7})
        self.assertEqual(resolution["status"], "resolved")
        self.assertEqual(resolution["source"], "state_action_schema_json.state_layout")

    def test_gripper_resolution_handles_agibot_dimension_blocks(self):
        metadata = {
            "state_action_schema_json": json.dumps({
                "state_dim": 20,
                "state_dims": [
                    "state/joint/position:14",
                    "state/effector/position:2",
                    "state/head/position:2",
                    "state/waist/position:2",
                ],
            })
        }
        indices, resolution = MODULE.resolve_gripper_indices(
            "auto", metadata, "state", 20,
        )
        self.assertEqual(indices, {14, 15})
        self.assertEqual(resolution["source"], "state_action_schema_json.state_dims")

    def test_gripper_resolution_uses_feature_names(self):
        metadata = {
            "state_action_schema_json": {
                "action_dim": 5,
                "action_feature": {
                    "names": ["joint_1", "left_gripper_open", "eef_x", "eef_y", "eef_z"],
                },
            }
        }
        indices, resolution = MODULE.resolve_gripper_indices(
            "auto", metadata, "action", 5,
        )
        self.assertEqual(indices, {1})
        self.assertEqual(
            resolution["source"], "state_action_schema_json.action_feature.names",
        )

    def test_gripper_resolution_pairs_top_level_fields_and_widths(self):
        metadata = {
            "state_fields_json": json.dumps(["joint", "gripper"]),
            "state_dims_json": json.dumps([7, 1]),
        }
        indices, resolution = MODULE.resolve_gripper_indices(
            "auto", metadata, "state", 8,
        )
        self.assertEqual(indices, {7})
        self.assertEqual(resolution["source"], "state_dims_json")

    def test_gripper_resolution_does_not_treat_eef_or_hand_as_gripper(self):
        metadata = {
            "state_action_schema_json": {
                "state_layout": ["left_hand_joint", "eef_x", "eef_y"],
            }
        }
        indices, resolution = MODULE.resolve_gripper_indices(
            "auto", metadata, "state", 3,
        )
        self.assertEqual(indices, set())
        self.assertEqual(resolution["status"], "resolved")

    def test_gripper_resolution_reports_unresolved_metadata(self):
        indices, resolution = MODULE.resolve_gripper_indices(
            "auto", {}, "state", 7,
        )
        self.assertEqual(indices, set())
        self.assertEqual(resolution["status"], "unresolved")

    def test_explicit_gripper_indices_are_range_checked(self):
        with self.assertRaisesRegex(ValueError, "outside dimension 2"):
            MODULE.resolve_gripper_indices("2", {}, "action", 2)

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

    def test_s3_preserves_nonfinite_gripper_flags(self):
        thresholds = {"lower": [-10.0, -10.0], "upper": [10.0, 10.0]}
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                result = MODULE.detect_s3(
                    np.array([[0.0, value]]), thresholds, {1},
                )
                self.assertTrue(result["flagged"])
                self.assertEqual(result["frames"], [0])
                self.assertEqual(result["hits"][0]["dim"], 1)
                self.assertEqual(result["hits"][0]["reason"], "nonfinite")

    def test_s3_excludes_inf_from_quantile_fit(self):
        values = np.array([[0.0], [1.0], [2.0], [np.inf]])
        episode = MODULE.Episode("fit", values, values.copy(), {})
        thresholds = MODULE.fit_s3([episode], MODULE.S3Config(), set(), set())
        state = thresholds["state"]
        self.assertEqual(state["finite_counts"], [3])
        self.assertEqual(state["invalid_threshold_dimensions"], [])
        self.assertTrue(np.isfinite(state["q01"][0]))
        self.assertTrue(np.isfinite(state["q99"][0]))

    def test_s3_reports_all_nonfinite_dimension(self):
        values = np.array([[np.nan], [np.inf], [-np.inf]])
        episode = MODULE.Episode("fit", values, values.copy(), {})
        thresholds = MODULE.fit_s3([episode], MODULE.S3Config(), set(), set())
        state = thresholds["state"]
        self.assertEqual(state["finite_counts"], [0])
        self.assertEqual(state["invalid_threshold_dimensions"], [0])

    def test_s3_samples_uniform_global_row_indices(self):
        episodes = []
        start = 0
        for length in (7, 11, 12):
            values = np.arange(start, start + length, dtype=float)[:, None]
            episodes.append(MODULE.Episode(str(start), values, values.copy(), {}))
            start += length
        expected_indices = np.sort(np.random.default_rng(0).choice(30, 10, replace=False))
        sampled, total_rows = MODULE._uniform_sample_rows(
            episodes, "state", 10, np.random.default_rng(0),
        )
        self.assertEqual(total_rows, 30)
        np.testing.assert_array_equal(sampled[:, 0], expected_indices)

    def test_streaming_reservoir_keeps_exact_global_priority_sample(self):
        values = np.arange(100, dtype=np.float32)[:, None]
        expected = np.argpartition(np.random.default_rng(7).random(100), 9)[:10]
        reservoir = MODULE._PriorityReservoir(capacity=10, columns=1, seed=7)
        reservoir.update(values[:7])
        reservoir.update(values[7:20])
        reservoir.update(values[20:])

        self.assertEqual(reservoir.sample.shape, (10, 1))
        np.testing.assert_array_equal(
            np.sort(reservoir.sample[:, 0].astype(int)), np.sort(expected),
        )


if __name__ == "__main__":
    unittest.main()
