# BiPlay to InfiData Conversion Guide

This document unifies the conversion entrypoints for BiPlay datasets in this workspace.

## 1) Conversion scripts

- Raw HDF5 converter:
  - `scripts/convert/convert_biplay_raw.py`
- TFRecord release converter:
  - `scripts/convert/convert_biplay_tfrecord.py`

Both scripts write the same InfiData structure:

- `data/biplay/chunk-000/episode_XXXXXX.parquet`
- `meta/episodes.jsonl`
- `meta/segments.jsonl`
- `meta/tasks.json`
- `meta/robots.json`
- `meta/stats.json`
- `videos/biplay/episode_XXXXXX_<camera>.mp4`

## 2) Raw HDF5 path (preferred when available)

Use this when a subset is available as raw HDF5 (for example `aloha_pen_uncap_diverse_raw`).

### Mini test

```bash
/mnt/data/szeluresearch/ELUBrain/.venv/bin/python scripts/convert/convert_biplay_raw.py \
  --raw_root /mnt/data/szeluresearch/ELUBrain/BiPlay/aloha_pen_uncap_diverse_raw \
  --out_root /mnt/data/szeluresearch/InfiData/examples/biplay_pen_uncap_mini \
  --num_episodes 3
```

### Full conversion

```bash
/mnt/data/szeluresearch/ELUBrain/.venv/bin/python scripts/convert/convert_biplay_raw.py \
  --raw_root /mnt/data/szeluresearch/ELUBrain/BiPlay/aloha_pen_uncap_diverse_raw \
  --out_root /mnt/data/szeluresearch/InfiData/examples/biplay_pen_uncap_full \
  --all_episodes
```

## 3) TFRecord release path

Use this when only the TFRecord package exists (for example `aloha_drawer_dataset`, `aloha_play_dataset`).

### Mini test

```bash
/mnt/data/szeluresearch/ELUBrain/.venv/bin/python scripts/convert/convert_biplay_tfrecord.py \
  --dataset_root /mnt/data/szeluresearch/ELUBrain/BiPlay/aloha_pick_place_dataset \
  --out_root /mnt/data/szeluresearch/InfiData/examples/biplay_pick_place_mini \
  --num_episodes 3
```

### Full conversion

```bash
/mnt/data/szeluresearch/ELUBrain/.venv/bin/python scripts/convert/convert_biplay_tfrecord.py \
  --dataset_root /mnt/data/szeluresearch/ELUBrain/BiPlay/aloha_pick_place_dataset \
  --out_root /mnt/data/szeluresearch/InfiData/examples/biplay_pick_place_full \
  --all_episodes
```

## 4) Suggested output naming

- Raw subset outputs:
  - `examples/biplay_<subset>_full`
  - `examples/biplay_<subset>_mini`
- TFRecord subset outputs:
  - `examples/biplay_<subset>_full`
  - `examples/biplay_<subset>_mini`

Examples:

- `examples/biplay_pen_uncap_full`
- `examples/biplay_dough_cut_full`
- `examples/biplay_drawer_full`
- `examples/biplay_play_full`

## 5) Important notes

- `subtask` segments are auto-generated placeholders and should be replaced by human or VLM-assisted labels.
- Video frames are decoded and re-encoded to mp4; this increases runtime and disk usage.
- The TFRecord converter uses low-level TFRecord reading, so it does not require TensorFlow runtime execution.
- `speed_bin`, `quality`, `success`, and `mistake` are initialized heuristically for pipeline compatibility.

## 6) Quick verification

After a run, verify:

```bash
find /mnt/data/szeluresearch/InfiData/examples/biplay_pick_place_full -maxdepth 3 -type f | head
cat /mnt/data/szeluresearch/InfiData/examples/biplay_pick_place_full/meta/stats.json
```
