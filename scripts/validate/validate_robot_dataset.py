import argparse
import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd


def to_native(value):
    if isinstance(value, np.ndarray):
        return [to_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [to_native(item) for item in value]
    if isinstance(value, dict):
        return {key: to_native(item) for key, item in value.items()}
    return value


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_dataset(dataset_root: Path, schema_root: Path):
    row_schema = json.loads((schema_root / "robot_episode.schema.json").read_text(encoding="utf-8"))
    segment_schema = json.loads(
        (schema_root / "segment_annotation.schema.json").read_text(encoding="utf-8")
    )

    parquet_files = sorted(dataset_root.glob("data/*/chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No InfiData episode parquet files under {dataset_root}")

    errors = []
    episode_summaries = []
    for parquet_path in parquet_files:
        df = pd.read_parquet(parquet_path)
        if df.empty:
            errors.append(f"{parquet_path}: empty parquet")
            continue

        for row_index, row in df.iterrows():
            record = {key: to_native(value) for key, value in row.items()}
            try:
                jsonschema.validate(record, row_schema)
            except jsonschema.ValidationError as exc:
                errors.append(f"{parquet_path}: row {row_index}: {exc.message}")
                break

        frame_index = df["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(frame_index, np.arange(len(df))):
            errors.append(f"{parquet_path}: frame_index is not contiguous")
        if not df["timestamp"].is_monotonic_increasing:
            errors.append(f"{parquet_path}: timestamp is not monotonic")
        if df["episode_index"].nunique() != 1:
            errors.append(f"{parquet_path}: multiple episode_index values")

        state_dim = {len(value) for value in df["observation.state"]}
        action_dim = {len(value) for value in df["action"]}
        if len(state_dim) != 1 or len(action_dim) != 1:
            errors.append(f"{parquet_path}: inconsistent state/action dimensions")

        for path_column in [name for name in df.columns if name.startswith("video.") and name.endswith(".path")]:
            for relative_path in df[path_column].dropna().unique():
                video_path = dataset_root / str(relative_path)
                if not video_path.exists():
                    errors.append(f"{parquet_path}: missing video {video_path}")

        episode_summaries.append(
            {
                "path": str(parquet_path.relative_to(dataset_root)),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "state_dim": next(iter(state_dim)),
                "action_dim": next(iter(action_dim)),
            }
        )

    segments_path = dataset_root / "meta" / "segments.jsonl"
    segments = load_jsonl(segments_path)
    for index, segment in enumerate(segments):
        try:
            jsonschema.validate(segment, segment_schema)
        except jsonschema.ValidationError as exc:
            errors.append(f"{segments_path}: record {index}: {exc.message}")

    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    episodes = load_jsonl(episodes_path)
    if len(episodes) != len(parquet_files):
        errors.append(
            f"{episodes_path}: {len(episodes)} records for {len(parquet_files)} parquet files"
        )

    return {
        "valid": not errors,
        "dataset_root": str(dataset_root),
        "num_episodes": len(parquet_files),
        "num_segments": len(segments),
        "episodes": episode_summaries,
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument(
        "--schema_root",
        default=str(Path(__file__).resolve().parents[2] / "schemas"),
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    result = validate_dataset(
        dataset_root=Path(args.dataset_root).resolve(),
        schema_root=Path(args.schema_root).resolve(),
    )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)

    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(text + "\n", encoding="utf-8")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
