"""Load experiment CSVs (URL with local cache, optional local-only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

AB_URL = (
    "https://raw.githubusercontent.com/jemc36/Udacity-DAND-AB-test-ecommerce/"
    "master/ab_data.csv"
)
COUNTRIES_URL = (
    "https://raw.githubusercontent.com/jemc36/Udacity-DAND-AB-test-ecommerce/"
    "master/countries.csv"
)
REPO_URL = "https://github.com/jemc36/Udacity-DAND-AB-test-ecommerce"


def project_root_from(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    if p.name == "notebooks":
        return p.parent
    return p


def load_ab_and_countries(
    data_dir: Path | None = None,
    *,
    prefer_local: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (ab_data, countries, load_path_description)."""
    root = project_root_from()
    data_dir = data_dir or (root / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    ab_path = data_dir / "ab_data.csv"
    cty_path = data_dir / "countries.csv"

    if prefer_local and ab_path.exists() and cty_path.exists():
        return (
            pd.read_csv(ab_path),
            pd.read_csv(cty_path),
            f"local cache: {ab_path} + {cty_path}",
        )

    try:
        ab = pd.read_csv(AB_URL)
        cty = pd.read_csv(COUNTRIES_URL)
        ab.to_csv(ab_path, index=False)
        cty.to_csv(cty_path, index=False)
        return ab, cty, f"raw GitHub URLs (cached under {data_dir})"
    except Exception as err:
        clone_dir = data_dir / "Udacity-DAND-AB-test-ecommerce"
        if not (clone_dir / "ab_data.csv").exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)],
                check=True,
            )
        ab = pd.read_csv(clone_dir / "ab_data.csv")
        cty = pd.read_csv(clone_dir / "countries.csv")
        return ab, cty, f"git clone fallback ({err!r}) at {clone_dir}"
