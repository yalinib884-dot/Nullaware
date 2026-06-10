"""
backend/profiling/generate_profile.py
Generates ydata-profiling HTML and JSON reports from a CSV file.
"""
import json
import pandas as pd
from pathlib import Path
from ydata_profiling import ProfileReport
import logging

logger = logging.getLogger(__name__)


def generate_profile(csv_path: str | Path, output_dir: str | Path) -> dict:
    """
    Generate profile.html and profile.json from a CSV file.

    Args:
        csv_path: Path to the uploaded CSV file.
        output_dir: Directory where profile files will be stored.

    Returns:
        dict with keys: html_path, json_path, dataframe_shape
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = csv_path.stem
    html_path = output_dir / f"{stem}_profile.html"
    json_path = output_dir / f"{stem}_profile.json"

    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    logger.info("Generating ydata-profiling report...")
    profile = ProfileReport(
        df,
        title=f"NulAware AI Profile: {stem}",
        explorative=True,
        minimal=False,
        correlations={
            "pearson": {"calculate": True},
            "spearman": {"calculate": True},
        },
    )

    logger.info(f"Saving HTML report to {html_path}")
    profile.to_file(html_path)

    logger.info(f"Saving JSON report to {json_path}")
    profile_json = json.loads(profile.to_json())
    with open(json_path, "w") as f:
        json.dump(profile_json, f, indent=2, default=str)

    logger.info("Profile generation complete.")
    return {
        "html_path": str(html_path),
        "json_path": str(json_path),
        "dataframe_shape": df.shape,
        "columns": list(df.columns),
    }