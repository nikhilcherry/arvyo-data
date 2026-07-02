#!/usr/bin/env python
"""FITS -> detrended .npz. See CLAUDE_CODE_DATA_HANDOUT.md Phase 3."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import wotan
from astropy.io import fits
from astropy.stats import sigma_clip
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

WINDOW_LENGTH_DAYS = 0.5
SIGMA_UPPER = 5.0
# wotan biweight w/ a 0.5 d window partially flattens short-Prot rotation
# signals, so starspot/null also keep an un-detrended flux_raw array.
UNDETRENDED_LABELS = {"starspot", "null"}


def get_crowdsap(fits_path):
    try:
        with fits.open(fits_path) as hdul:
            for hdu in hdul:
                if "CROWDSAP" in hdu.header:
                    return float(hdu.header["CROWDSAP"]), False
    except Exception:
        pass
    return 1.0, True  # missing on some TESS-SPOC FFI products; default + flag


def load_light_curve(fits_path):
    import lightkurve as lk
    try:
        lc = lk.read(str(fits_path), flux_column="pdcsap_flux")
    except Exception:
        lc = lk.read(str(fits_path))
    return lc


def preprocess_one(tic_id, label, fits_path, meta_row):
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

    if time.size < 50:
        raise ValueError(f"too few finite cadences ({time.size})")

    order = np.argsort(time)
    time, flux, flux_err = time[order], flux[order], flux_err[order]

    # sigma-clip outliers upward only; downward dips are the transit signal
    clipped = sigma_clip(flux, sigma_upper=SIGMA_UPPER, sigma_lower=np.inf, masked=True)
    keep = ~clipped.mask
    time, flux, flux_err = time[keep], flux[keep], flux_err[keep]

    median_raw = np.nanmedian(flux)
    flux_raw_norm = flux / median_raw  # normalized, un-detrended

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
        "tic_id": tic_id,
        "label": label,
        "sector": getattr(lc, "sector", None) or -1,
        "period_days": meta_row.get("period_days", np.nan),
        "epoch_btjd": meta_row.get("epoch_btjd", np.nan),
        "crowdsap": crowdsap,
        "crowdsap_missing": crowdsap_missing,
    }
    if label in UNDETRENDED_LABELS:
        out["flux_raw"] = flux_raw_norm

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "manifest.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    # keep_default_na=False: pandas' default NA sentinels include the literal
    # string "null", which is one of our label values (the quiet-star class).
    manifest = pd.read_csv(args.manifest, keep_default_na=False, na_values=[""])
    if args.label:
        manifest = manifest[manifest["label"] == args.label]
    if args.limit:
        manifest = manifest.head(args.limit)

    ok, failed, missing_raw = 0, 0, 0
    failed_rows = []

    pbar = tqdm(manifest.itertuples(index=False), total=len(manifest), desc="preprocessing")
    for row in pbar:
        row_dict = row._asdict()
        tic_id, label = row_dict["tic_id"], row_dict["label"]
        fits_path = RAW_DIR / f"{tic_id}.fits"
        out_path = PROC_DIR / label / f"{tic_id}.npz"
        if out_path.exists():
            ok += 1
            continue
        if not fits_path.exists():
            missing_raw += 1
            continue  # not downloaded yet; verify_dataset.py reports coverage gaps

        try:
            result = preprocess_one(tic_id, label, fits_path, row_dict)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path, **result)
            ok += 1
        except Exception as e:
            failed += 1
            failed_rows.append({"tic_id": tic_id, "label": label, "reason": str(e)})
        pbar.set_postfix(ok=ok, failed=failed, missing_raw=missing_raw)

    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(ROOT / "data" / "failed_preprocess.csv", index=False)

    print(f"\npreprocessed ok={ok} failed={failed} missing_raw_fits={missing_raw}")


if __name__ == "__main__":
    main()
