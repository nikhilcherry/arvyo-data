#!/usr/bin/env python
"""Select the primary candidate for a real Kepler DR25 centroid-offset blend
sample, per the pre-registered filter below (do not adjust after seeing
results -- see task handout "Scientific-honesty hard requirement").

Source: cumulative KOI table (same TAP source/cache as
04_build_kepler_manifest.py), FALSE POSITIVE dispositions with the centroid
offset flag set (koi_fpflag_co == 1) -- catalog-confirmed blends where a
background eclipsing binary was resolved by centroid shift.

Filter, in order:
  1. koi_disposition == "FALSE POSITIVE" and koi_fpflag_co == 1
  2. koi_fpflag_nt == 0            (signal IS transit-like, just off-target)
  3. 0.5 <= koi_period <= 5.0 days (many events per quarter)
  4. koi_depth >= 500 ppm          (detectable after dilution)
  5. koi_model_snr >= 20           (strong catalog-level detection)

Survivors are ranked by koi_model_snr descending, tie-broken by kepid
ascending (stable sort) -- deterministic, no randomness. --seed is accepted
only for interface consistency with the other scripts/NN.py in this repo.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
CATALOG_DIR = ROOT / "data" / "kepler" / "catalogs"
KOI_CUMULATIVE_PATH = CATALOG_DIR / "koi_cumulative.csv"
OUT_PATH = ROOT / "data" / "catalogs" / "blend_candidates.csv"

TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={q}&format=csv"

FLAG_COLS = ["koi_fpflag_co", "koi_fpflag_ss", "koi_fpflag_nt", "koi_fpflag_ec"]


def fetch_koi_cumulative() -> pd.DataFrame:
    """Load the cached DR25 KOI cumulative table, or fetch it via TAP."""
    if KOI_CUMULATIVE_PATH.exists():
        print(f"  using cached {KOI_CUMULATIVE_PATH}")
        return pd.read_csv(KOI_CUMULATIVE_PATH)

    print("  no cached koi_cumulative.csv, fetching cumulative KOI table via TAP...")
    url = TAP_BASE.format(q="select+*+from+cumulative")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    text = r.text
    if text.lstrip().startswith("<?xml") or "VOTABLE" in text[:200]:
        raise RuntimeError(f"TAP returned an error document: {text[:300]}")
    df = pd.read_csv(pd.io.common.StringIO(text))
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(KOI_CUMULATIVE_PATH, index=False)
    print(f"  fetched {len(df)} rows -> {KOI_CUMULATIVE_PATH}")
    return df


def flags_string(row) -> str:
    active = [c.replace("koi_fpflag_", "") for c in FLAG_COLS if row[c] == 1]
    return "+".join(active) if active else "none"


def select_candidates(koi: pd.DataFrame) -> pd.DataFrame:
    df = koi.copy()
    for c in FLAG_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["koi_period"] = pd.to_numeric(df["koi_period"], errors="coerce")
    df["koi_depth"] = pd.to_numeric(df["koi_depth"], errors="coerce")
    df["koi_model_snr"] = pd.to_numeric(df["koi_model_snr"], errors="coerce")

    disp = df["koi_disposition"].astype(str).str.strip().str.upper()

    mask = (
        (disp == "FALSE POSITIVE")
        & (df["koi_fpflag_co"] == 1)
        & (df["koi_fpflag_nt"] == 0)
        & (df["koi_period"].between(0.5, 5.0))
        & (df["koi_depth"] >= 500)
        & (df["koi_model_snr"] >= 20)
    )
    survivors = df.loc[mask].copy()
    survivors["flags"] = survivors.apply(flags_string, axis=1)

    # Deterministic ranking: snr desc, tie-break kepid asc. mergesort is
    # stable, matters if snr ties ever occur across a re-fetch.
    survivors = survivors.sort_values(
        by=["koi_model_snr", "kepid"], ascending=[False, True], kind="mergesort"
    )

    out = survivors[[
        "kepid", "kepoi_name", "koi_period", "koi_depth", "koi_model_snr", "flags",
    ]].rename(columns={
        "kepid": "kic",
        "kepoi_name": "koi",
        "koi_period": "period_days",
        "koi_depth": "depth_ppm",
        "koi_model_snr": "snr",
    })
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Unused -- selection is fully deterministic (no sampling). "
             "Accepted only for CLI consistency with the other scripts/NN.py.",
    )
    ap.add_argument("--top-n", type=int, default=5)
    args = ap.parse_args()

    koi = fetch_koi_cumulative()
    if koi is None or koi.empty:
        print("ERROR: no cumulative KOI table available.", file=sys.stderr)
        sys.exit(1)

    ranked = select_candidates(koi)
    print(f"\n{len(ranked)} candidates pass the pre-registered filter (of {len(koi)} KOI rows)")

    top = ranked.head(args.top_n)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(OUT_PATH, index=False)
    print(f"Top {len(top)} written to {OUT_PATH}\n")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
