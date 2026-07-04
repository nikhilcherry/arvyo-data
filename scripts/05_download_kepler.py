#!/usr/bin/env python
"""Download one Kepler FITS quarter per manifest target via lightkurve/MAST.
Resumable, retries with backoff, --limit/--label flags for testing.
See Kepler addendum Phase A2.

Long cadence (30-min) is used -- DR25 vetting was done on it, and it keeps
download size down. One quarter (the first available) is downloaded per
target; Kepler quarters have very different lengths (Q1 ~33d, Q4 has a dead
module gap) but one-quarter-per-target is still fine for a transfer set.

A KIC can host multiple KOIs (manifest has one row per KOI), so targets are
deduped by KIC id before downloading -- multi-KOI stars would otherwise be
fetched (and skip-checked) redundantly.
"""
import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "kepler" / "raw"
FAILED_PATH = ROOT / "data" / "kepler" / "failed_downloads.csv"

RETRIES = 3
# The TESS download (02_download_lightcurves.py) may be running concurrently;
# be more bandwidth-polite than the TESS script's 0.5s.
SLEEP_BETWEEN = 1.0


def _search_and_download(kic_id, dest_path):
    import lightkurve as lk

    last_err = None
    for attempt in range(RETRIES):
        try:
            sr = lk.search_lightcurve(f"KIC {kic_id}", author="Kepler", cadence="long")
            if len(sr) == 0:
                return False, "no_results"
            # Quarter 00 is a ~10-day commissioning run that sorts first but
            # yields too few cadences after quality filtering (see
            # MIN_CADENCES in 06_preprocess_kepler.py); prefer a full quarter
            # when one is available.
            missions = list(sr.table["mission"])
            pick = next((i for i, m in enumerate(missions) if "Quarter 00" not in m), 0)
            with tempfile.TemporaryDirectory() as tmpdir:
                sr[pick].download(download_dir=tmpdir)
                fits_files = list(Path(tmpdir).rglob("*.fits"))
                if not fits_files:
                    return False, "download_no_fits"
                fits_path = max(fits_files, key=lambda p: p.stat().st_mtime)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(fits_path, dest_path)
            return True, None
        except Exception as e:
            last_err = str(e)
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)
    return False, f"error: {last_err}"


def download_target(kic_id):
    dest_path = RAW_DIR / f"{kic_id}.fits"
    if dest_path.exists():
        return "skipped", None

    ok, reason = _search_and_download(kic_id, dest_path)
    if ok:
        return "kepler_lc", None
    return "failed", reason


def verify_fits_readable(path):
    import lightkurve as lk
    try:
        lk.read(str(path))
        return True
    except Exception as e:
        print(f"  WARNING: {path} failed to open with lightkurve.read(): {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "kepler_manifest.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default=None,
                     help="restrict to one manifest label, e.g. planet|eb|blend|unknown")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, keep_default_na=False, na_values=[""])
    if args.label:
        manifest = manifest[manifest["label"] == args.label]

    kic_ids = manifest["tic_id"].dropna().astype(int).drop_duplicates().tolist()
    if args.limit:
        kic_ids = kic_ids[: args.limit]

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)

    counts = {"kepler_lc": 0, "skipped": 0, "failed": 0}
    failed_rows = []

    pbar = tqdm(kic_ids, desc="downloading kepler")
    for kic_id in pbar:
        status, reason = download_target(kic_id)
        counts[status] = counts.get(status, 0) + 1
        if status == "failed":
            failed_rows.append({"kic_id": kic_id, "reason": reason})
        pbar.set_postfix(counts)
        if status != "skipped":
            time.sleep(SLEEP_BETWEEN)

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        if FAILED_PATH.exists():
            existing = pd.read_csv(FAILED_PATH)
            failed_df = pd.concat([existing, failed_df], ignore_index=True)
            failed_df = failed_df.drop_duplicates(subset="kic_id", keep="last")
        failed_df.to_csv(FAILED_PATH, index=False)

    print("\nDownload summary:")
    for k, v in counts.items():
        print(f"  {k:14s} {v}")
    if failed_rows:
        print(f"  logged {len(failed_rows)} new failures -> {FAILED_PATH}")

    if args.limit:
        print("\nVerifying downloaded files open with lightkurve.read()...")
        ok_count = 0
        for kic_id in kic_ids:
            path = RAW_DIR / f"{kic_id}.fits"
            if path.exists() and verify_fits_readable(path):
                ok_count += 1
        print(f"  {ok_count}/{len(kic_ids)} files verified readable")
        print("\nThis was a --limit run. Ask the user before starting the full "
              "download (~4,100 quarters, a couple of GB, 1-2 hours).")


if __name__ == "__main__":
    main()
