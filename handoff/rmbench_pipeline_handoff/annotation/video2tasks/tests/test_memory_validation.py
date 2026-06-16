import unittest

from video2tasks.server.windowing import Window, build_memory_segments_via_cuts
from video2tasks.validation import validate_memory_annotation, validate_memory_coverage


class MemoryValidationTest(unittest.TestCase):
    def test_valid_annotation(self):
        annotation = {
            "thought": "The memory changes once.",
            "transitions": [6],
            "summaries": ["before", "after"],
            "change_event_types": [["initial_observation"], ["memory_updated"]],
        }
        self.assertIsNone(validate_memory_annotation(annotation, 16))

    def test_rejects_missing_summary(self):
        annotation = {
            "thought": "The memory changes once.",
            "transitions": [6],
            "summaries": ["after"],
            "change_event_types": [["memory_updated"]],
        }
        self.assertIn("exactly 2", validate_memory_annotation(annotation, 16))

    def test_rejects_invalid_transition(self):
        annotation = {
            "thought": "The memory changes once.",
            "transitions": [0],
            "summaries": ["before", "after"],
            "change_event_types": [["initial_observation"], ["memory_updated"]],
        }
        self.assertIn("between 1 and 15", validate_memory_annotation(annotation, 16))

    def test_rejects_coverage_gap(self):
        records = [
            {"start_frame": 0, "end_frame": 4, "summary": "before"},
            {"start_frame": 6, "end_frame": 9, "summary": "after"},
        ]
        self.assertIn("frame 5", validate_memory_coverage(records, 10))

    def test_short_segment_is_preserved(self):
        frame_ids = [0, 9, 19, 29, 39, 49, 58, 68, 78, 88, 98, 107, 117, 127, 137, 147]
        windows = [Window(0, 0, 147, frame_ids)]
        by_wid = {
            0: {
                "vlm_json": {
                    "transitions": [1],
                    "summaries": ["before", "after"],
                    "change_event_types": [["initial_observation"], ["memory_updated"]],
                }
            }
        }

        result = build_memory_segments_via_cuts("episode", windows, by_wid, 30.0, 148)
        records = [
            {
                "start_frame": seg["start_frame"],
                "end_frame": seg["end_frame"] - 1,
                "summary": seg["summary"],
            }
            for seg in result["memory_segments"]
        ]
        self.assertIsNone(validate_memory_coverage(records, 148))


if __name__ == "__main__":
    unittest.main()
