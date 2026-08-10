#!/usr/bin/env python3

from __future__ import annotations

import mmap
import tempfile
import unittest
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from materialize_filtered_rlds import _filter_shard, extract_episode_key, iter_tfrecord


def _example(index: int, global_key: str = "") -> bytes:
    features = {
        "episode_metadata/episode_index": tf.train.Feature(
            int64_list=tf.train.Int64List(value=[index])
        ),
        "large_ignored_payload": tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[b"x" * 100_000])
        ),
    }
    if global_key:
        features["episode_metadata/global_episode_key"] = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[global_key.encode()])
        )
    return tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()


class MaterializeFilteredRldsTest(unittest.TestCase):
    def test_extract_key_prefers_global_key(self) -> None:
        raw = _example(7, "shard:7")
        view = memoryview(raw)
        try:
            self.assertEqual(extract_episode_key(view, 0, len(view), 99), "shard:7")
        finally:
            view.release()

    def test_filter_shard_copies_only_kept_serialized_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.tfrecord", root / "output.tfrecord"
            with tf.io.TFRecordWriter(str(source)) as writer:
                # The older decision uses episode_index even though the raw
                # record also contains a global key (as in AgiBot).
                writer.write(_example(1, "namespace:1"))
                writer.write(_example(2))
            decisions = {
                "1": {"episode": "1", "delete": True},
                "2": {"episode": "2", "delete": False},
            }
            with tqdm(disable=True) as progress:
                state = _filter_shard(
                    source,
                    output,
                    root / "checkpoint.json",
                    decisions,
                    "test-sha",
                    "episode_index",
                    0,
                    progress,
                )
            self.assertEqual((state["source_records"], state["deleted"], state["kept"]), (2, 1, 1))
            with output.open("rb") as handle, mmap.mmap(
                handle.fileno(), 0, access=mmap.ACCESS_READ
            ) as mapped:
                view = memoryview(mapped)
                try:
                    records = list(iter_tfrecord(view))
                    self.assertEqual(len(records), 1)
                    _, start, end, _ = records[0]
                    self.assertEqual(extract_episode_key(view, start, end, 0), "2")
                finally:
                    view.release()


if __name__ == "__main__":
    unittest.main()
