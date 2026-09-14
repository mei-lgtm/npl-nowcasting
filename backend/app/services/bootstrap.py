"""Bootstrap demo data on startup."""

from pathlib import Path

from app.core.config import settings
from app.data.synthetic import write_demo_files


def bootstrap_demo_if_needed() -> None:
    data_dir = Path(settings.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[2] / data_dir
    obs = data_dir / "observations.parquet"
    if settings.app_mode == "demo" and not obs.exists():
        write_demo_files(data_dir)
