# Observe and Pickup Memory Annotation

This run annotates long-term memory summaries for the 50 RMBench
`observe_and_pickup` episodes. Existing human-labeled subtasks are used as
context and are not modified.

## Paths

- Code: `/mnt/workspace/wudi/InfiData-personal/video2tasks`
- Dataset: `/mnt/workspace/InfiData/Rmbench_infi/observe_and_pickup`
- Config: `config.observe_and_pickup.yaml`
- Run output: `runs/observe_and_pickup/observe_and_pickup_memory_v1`

The dataset already contains memory annotations and parquet `summary` values.
The production config writes back and will replace them episode by episode.

## 1. Install

```bash
cd /mnt/workspace/wudi/InfiData-personal/video2tasks
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install "numpy>=1.24,<2"
python -m pip install -e . --no-deps --no-build-isolation

python -c "import video2tasks; print(video2tasks.__file__)"
which v2t-server
which v2t-worker
which v2t-validate
```

The imported package must point into this directory, not the copy under
`/mnt/data/szeluresearch`.

This machine already provides PyTorch, Transformers, Accelerate,
FlashAttention, Pandas, PyArrow, and OpenCV through its system Python. The
NumPy 1.x constraint is intentional: the current SciPy build is not compatible
with NumPy 2.x, and that incompatibility prevents `transformers` from importing
Qwen3-VL.

Before starting the worker, verify that the shell can see a GPU:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

The configured model is `Qwen/Qwen3-VL-8B-Instruct`. If it is not already in
the Hugging Face cache, the worker needs network access to download it, or
`worker.qwen3vl.model_path` must be changed to an existing local model path.

## 2. Preflight

```bash
cd /mnt/workspace/wudi/InfiData-personal/video2tasks
source .venv/bin/activate
python scripts/preflight_observe_and_pickup.py
v2t-validate --config config.observe_and_pickup.yaml
```

Expected dataset count: 50 episodes.

## 3. Back Up Existing Results

Choose a new backup directory for every production run.

```bash
DATASET=/mnt/workspace/InfiData/Rmbench_infi/observe_and_pickup
BACKUP=/mnt/workspace/wudi/InfiData-personal/video2tasks/backups/observe_and_pickup_before_memory_v1

mkdir -p "$BACKUP"
cp "$DATASET/meta/memory_segments.jsonl" "$BACKUP/"
cp -a "$DATASET/data" "$BACKUP/"
```

Do not skip the parquet backup when
`update_parquet_memory_summaries: true`.

## 4. Start the Server

Terminal 1:

```bash
cd /mnt/workspace/wudi/InfiData-personal/video2tasks
source .venv/bin/activate
v2t-server --config config.observe_and_pickup.yaml
```

The first run should report:

```text
[Resume] Already done: 0/50 (computed_total=50)
```

If it reports `50/50`, change `run.run_id` in the config.

## 5. Start the Worker

Terminal 2:

```bash
cd /mnt/workspace/wudi/InfiData-personal/video2tasks
source .venv/bin/activate
v2t-worker --config config.observe_and_pickup.yaml
```

Expected worker messages include:

```text
[Worker] Annotation targets: ['memory']
[Done] ... Memory cuts: [...]
```

## 6. Monitor

```bash
RUN=/mnt/workspace/wudi/InfiData-personal/video2tasks/runs/observe_and_pickup/observe_and_pickup_memory_v1

find "$RUN" -name .DONE | wc -l
find "$RUN" -name windows.jsonl | wc -l
find "$RUN" -name '*annotated.mp4' | wc -l
```

All three counts should reach 50.

## 7. Inspect Results

```bash
RUN=/mnt/workspace/wudi/InfiData-personal/video2tasks/runs/observe_and_pickup/observe_and_pickup_memory_v1
DATASET=/mnt/workspace/InfiData/Rmbench_infi/observe_and_pickup

head -n 5 "$RUN/memory_segments.jsonl"
head -n 5 "$DATASET/meta/memory_segments.jsonl"
find "$RUN/visualizations" -name '*.mp4' | head
```

Check a parquet:

```bash
python - <<'PY'
import pandas as pd

p = "/mnt/workspace/InfiData/Rmbench_infi/observe_and_pickup/data/rmbench/chunk-000/episode_000000.parquet"
df = pd.read_parquet(p)
print(df[["frame_index", "subtask", "summary"]].to_string(index=False))
PY
```

Review summaries for future-information leakage, incorrect target identity,
empty text, and hallucinated object state before accepting the run.

## Resume and Rerun

- Restarting with the same `run_id` resumes from `windows.jsonl` and `.DONE`.
- A changed model, prompt, or annotation setting requires a new `run_id`.
- Server and worker must use the same port and config file.
