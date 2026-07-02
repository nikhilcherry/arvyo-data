#!/usr/bin/env python
"""Kepler FITS -> detrended .npz, same recipe and schema as 03_preprocess.py
(PDCSAP, quality==0, upward-only 5-sigma clip, wotan biweight 0.5d, median-
normalize), plus mission="kepler" metadata. See Kepler addendum Phase A3.

Deltas from the TESS preprocessor:
  - Kepler long cadence (30-min) means even full quarters have far fewer
    cadences than TESS's 2-min data; the too-few-cadences floor is lowered
    from TESS's threshold to 500 so short quarters (e.g. Q1) aren't dropped.
  - CROWDSAP is present in Kepler FITS headers too (same lookup as TESS).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wotan
from astropy.io import fits
from astropy.stats import sigma_clip
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "kepler" / "raw"
PROC_DIR = ROOT / "data" / "kepler" / "processed"

WINDOW_LENGTH_DAYS = 0.5
SIGMA_UPPER = 5.0
MIN_CADENCES = 500  # lowered from TESS's floor: Kepler LC quarters are shorter


def get_crowdsap(fits_path):
    try:
        with fits.open(fits_path) as hdul:
            for hdu in hdul:
                if "CROWDSAP" in hdu.header:
                    return float(hdu.header["CROWDSAP"]), False
    except Exception:
        pass
    return 1.0, True


def load_light_curve(fits_path):
    import lightkurve as lk
    try:
        lc = lk.read(str(fits_path), flux_column="pdcsap_flux")
    except Exception:
        lc = lk.read(str(fits_path))
    return lc


def preprocess_one(kic_id, label, fits_path, meta_row):
    lc = load_light_curve(fits_path)

    if hasattr(lc, "quality") and lc.quality is not None:
        lc = lc[np.asarray(lc.quality) == 0]

    time = np.asarray(lc.time.value, dtype=np.float64)
    flux = np.asarray(lc.flux.value, dtype=np.float64)
    if lc.flux_err is not None:
        flux_err = np.asarray(lc.flux_err.value, dtype=np.float64)
    else:
        flux_err = np.full_like(flux, np.nan)

    good = np.isfinite(time) & np.isfinite(flux)
    time, flux, flux_err = time[good], flux[good], flux_err[good]
    if flux_err.size and np.isfinite(flux_err).any():
        flux_err = np.where(np.isfinite(flux_err), flux_err, np.nanmedian(flux_err))
    else:
        flux_err = np.zeros_like(flux)

    if time.size < MIN_CADENCES:
        raise ValueError(f"too few finite cadences ({time.size} < {MIN_CADENCES})")

    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]

    clipped = sigma_clip(flux, sigma_upper=SIGMA_UPPER, sigma_lower=np.inf, masked=True)
    keep = ~clipped.mask
    time, flux, flux_err = time[keep], flux[keep], flux_err[keep]

    median_raw = np.nanmedian(flux)
    flux_raw_norm = flux / median_raw

    flattened_flux, trend = wotan.flatten(
        time, flux, method="biweight", window_length=WINDOW_LENGTH_DAYS,
        return_trend=True,
    )
    flux_err_detrended = flux_err / trend

    median_final = np.nanmedian(flattened_flux)
    flux_norm = flattened_flux / median_final
    flux_err_norm = flux_err_detrended / median_final

    crowdsap, crowdsap_missing = get_crowdsap(fits_path)

    out = {
        "time": time,
        "flux": flux_norm,
        "flux_err": flux_err_norm,
        "tic_id": kic_id,
        "label": label,
        "sector": getattr(lc, "quarter", None) or -1,
        "period_days": meta_row.get("period_days", np.nan),
        "epoch_btjd": meta_row.get("epoch_btjd", np.nan),
        "crowdsap": crowdsap,
        "crowdsap_missing": crowdsap_missing,
        "mission": "kepler",
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "kepler_manifest.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, keep_default_na=False, na_values=[""])
    if args.label:
        manifest = manifest[manifest["label"] == args.label]
    # dedupe by KIC: multiple KOI rows can share a kepid; the .npz is keyed
    # by kepid, so re-preprocessing the same star per KOI row is wasted work.
    manifest = manifest.drop_duplicates(subset="tic_id")
    if args.limit:
        manifest = manifest.head(args.limit)

    ok, failed, missing_raw = 0, 0, 0
    failed_rows = []

    pbar = tqdm(manifest.itertuples(index=False), total=len(manifest), desc="preprocessing kepler")
    for row in pbar:
        row_dict = row._asdict()
        kic_id, label = row_dict["tic_id"], row_dict["label"]
        fits_path = RAW_DIR / f"{kic_id}.fits"
        out_path = PROC_DIR / label / f"{kic_id}.npz"
        if out_path.exists():
            ok += 1
            continue
        if not fits_path.exists():
            missing_raw += 1
            continue

        try:
            result = preprocess_one(kic_id, label, fits_path, row_dict)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, **result)
            ok += 1
        except Exception as e:
            failed += 1
            failed_rows.append({"kic_id": kic_id, "label": label, "reason": str(e)})
        pbar.set_postfix(ok=ok, failed=failed, missing_raw=missing_raw)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(ROOT / "data" / "kepler" / "failed_preprocess.csv", index=False)

    print(f"\npreprocessed ok={ok} failed={failed} missing_raw_fits={missing_raw}")


if __name__ == "__main__":
    main()
