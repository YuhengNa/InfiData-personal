"""Config validation CLI entrypoint."""

import sys
import click
from pathlib import Path
from ..config import Config


@click.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to configuration file to validate"
)
def main(config: Path) -> None:
    """Validate a configuration file."""
    try:
        cfg = Config.from_yaml(config)
        click.echo(f"Configuration valid: {config}")
        click.echo(f"  Datasets: {len(cfg.datasets)}")
        for ds in cfg.datasets:
            click.echo(f"    - {ds.subset}: {ds.root} ({ds.format}, video_key={ds.video_key})")
        click.echo(f"  Run base: {cfg.run.base_dir}")
        click.echo(f"  Server: {cfg.server.host}:{cfg.server.port}")
        click.echo(f"  Worker backend: {cfg.worker.backend}")
        click.echo(f"  Annotation targets: {cfg.annotation.targets}")
        click.echo(f"  Segmentation mode: {cfg.segmentation.mode}")
        click.echo(f"  InfiData write back: {cfg.infidata.write_back}")
        click.echo(f"  Update parquet subtasks: {cfg.infidata.update_parquet_subtasks}")
        click.echo(f"  Update parquet memory summaries: {cfg.infidata.update_parquet_memory_summaries}")
        click.echo(f"  Parquet memory column: {cfg.infidata.parquet_memory_column}")
        click.echo(f"  Memory uses subtask context: {cfg.memory.use_subtask_context}")
        click.echo(f"  Memory aligns to subtasks: {cfg.memory.align_to_subtasks}")
        click.echo(f"  Visualization enabled: {cfg.visualization.enabled}")
        click.echo(f"  Windowing: {cfg.windowing.frames_per_window} frames per window")
        sys.exit(0)
    except Exception as e:
        click.echo(f"Configuration invalid: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
