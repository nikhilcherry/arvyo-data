#!/usr/bin/env python
"""Download one light curve FITS per manifest target via lightkurve/MAST.
Resumable, retries with backoff, --limit/--label flags for testing.
See CLAUDE_CODE_DATA_HANDOUT.md Phase 2.
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
RAW_DIR = ROOT / "data" / "raw"
FAILED_PATH = ROOT / "data" / "failed_downloads.csv"

RETRIES = 3
SLEEP_BETWEEN = 0.5


def _search_and_download(tic_id, author, cadence, dest_path):
    import lightkurve as lk

    kwargs = {"author": author}
    if cadence:
        kwargs["cadence"] = cadence

    last_err = None
    for attempt in range(RETRIES):
        try:
            sr = lk.search_lightcurve(f"TIC {tic_id}", **kwargs)
            if len(sr) == 0:
                return False, "no_results"
            with tempfile.TemporaryDirectory() as tmpdir:
                sr[0].download(download_dir=tmpdir)
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


def download_target(tic_id):
    dest_path = RAW_DIR / f"{tic_id}.fits"
    if dest_path.exists():
        return "skipped", None

    ok, reason = _search_and_download(tic_id, "SPOC", "short", dest_path)
    if ok:
        return "spoc_2min", None

    ok, reason = _search_and_download(tic_id, "TESS-SPOC", None, dest_path)
    if ok:
        return "tess_spoc_ffi", None

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
    ap.add_argument("--manifest", default=str(ROOT / "manifest.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default=None,
                     help="restrict to one manifest label, e.g. planet|eb|unknown|starspot|null")
    args = ap.parse_args()

    # keep_default_na=False: pandas' default NA sentinels include the literal
    # string "null", which is one of our label values (the quiet-star class).
    manifest = pd.read_csv(args.manifest, keep_default_na=False, na_values=[""])
    if args.label:
        manifest = manifest[manifest["label"] == args.label]
    if args.limit:
        manifest = manifest.head(args.limit)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)

    counts = {"spoc_2min": 0, "tess_spoc_ffi": 0, "skipped": 0, "failed": 0}
    failed_rows = []

    pbar = tqdm(manifest["tic_id"].tolist(), desc="downloading")
    for tic_id in pbar:
        status, reason = download_target(tic_id)
        counts[status] = counts.get(status, 0) + 1
        if status == "failed":
            failed_rows.append({"tic_id": tic_id, "reason": reason})
        pbar.set_postfix(counts)
        if status != "skipped":
            time.sleep(SLEEP_BETWEEN)

    if failed_rows:
        failed_df = pd.DataFrame(failed_rows)
        if FAILED_PATH.exists():
            existing = pd.read_csv(FAILED_PATH)
            failed_df = pd.concat([existing, failed_df], ignore_index=True)
            failed_df = failed_df.drop_duplicates(subset="tic_id", keep="last")
        failed_df.to_csv(FAILED_PATH, index=False)

    print("\nDownload summary:")
    for k, v in counts.items():
        print(f"  {k:14s} {v}")
    if failed_rows:
        print(f"  logged {len(failed_rows)} new failures -> {FAILED_PATH}")

    if args.limit:
        print("\nVerifying downloaded files open with lightkurve.read()...")
        ok_count = 0
        for tic_id in manifest["tic_id"].tolist():
            path = RAW_DIR / f"{tic_id}.fits"
            if path.exists() and verify_fits_readable(path):
                ok_count += 1
        print(f"  {ok_count}/{len(manifest)} files verified readable")
        print("\nThis was a --limit run. Ask the user before starting the full "
              "download (it may take 1-3 hours).")


if __name__ == "__main__":
    main()
