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
            click.echo(f"    - {ds.subset}: {ds.root}")
        click.echo(f"  Run base: {cfg.run.base_dir}")
        click.echo(f"  Server: {cfg.server.host}:{cfg.server.port}")
        click.echo(f"  Worker backend: {cfg.worker.backend}")
        click.echo(f"  Windowing: {cfg.windowing.frames_per_window} frames per window")
        sys.exit(0)
    except Exception as e:
        click.echo(f"Configuration invalid: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()