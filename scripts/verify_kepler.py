#!/usr/bin/env python
"""Sanity checks + summary stats over data/kepler/processed/. Mirrors
verify_dataset.py; see Kepler addendum Phase A4."""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "kepler_manifest.csv"
PROC_DIR = ROOT / "data" / "kepler" / "processed"

SEED = 42
SPOT_CHECK_N = 5
MIN_CADENCES = 500  # matches the preprocess floor (lower than TESS's 1000)


def phase_fold(time, period, epoch=None):
    if epoch is None or not np.isfinite(epoch):
        epoch = time[0]
    return ((time - epoch) / period) % 1.0


def load_npz(path):
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


# --------------------------------------------------------------- coverage --
def check_coverage(manifest):
    print("=== Coverage vs manifest ===")
    missing_total = 0
    counts = {}
    for label, group in manifest.groupby("label"):
        label_dir = PROC_DIR / label
        have = {p.stem for p in label_dir.glob("*.npz")} if label_dir.exists() else set()
        want = set(group["tic_id"].astype(str))
        missing = want - have
        counts[label] = (len(have & want), len(want))
        missing_total += len(missing)
        print(f"  {label:10s} {len(have & want)}/{len(want)} present"
              f"{f'  ({len(missing)} missing)' if missing else ''}")
    print(f"  total missing: {missing_total}")
    return counts


# ------------------------------------------------------------ spot checks --
def spot_check(manifest):
    print("\n=== Random .npz spot checks (5 per class) ===")
    rng = random.Random(SEED)
    problems = []
    for label in manifest["label"].unique():
        label_dir = PROC_DIR / label
        files = sorted(label_dir.glob("*.npz")) if label_dir.exists() else []
        if not files:
            print(f"  {label:10s} no files to check")
            continue
        sample = rng.sample(files, min(SPOT_CHECK_N, len(files)))
        for f in sample:
            d = load_npz(f)
            flux = d["flux"]
            issues = []
            if np.isnan(flux).any() or np.isnan(d["time"]).any():
                issues.append("NaNs present")
            med = np.nanmedian(flux)
            if not np.isclose(med, 1.0, atol=0.05):
                issues.append(f"median flux {med:.3f} != 1.0")
            if len(flux) < MIN_CADENCES:
                issues.append(f"only {len(flux)} cadences (<{MIN_CADENCES})")
            if str(d.get("mission")) != "kepler":
                issues.append(f"mission={d.get('mission')} != kepler")
            if issues:
                problems.append((f.name, issues))
        print(f"  {label:10s} checked {len(sample)} files")
    if problems:
        print("  ISSUES:")
        for name, issues in problems:
            print(f"    {name}: {'; '.join(issues)}")
    else:
        print("  no issues found")
    return problems


# --------------------------------------------------------- phase-fold PNGs --
def pick_smoke_targets(manifest):
    targets = []

    planet_pool = manifest[manifest["label"] == "planet"]
    if len(planet_pool):
        targets.append(("planet", planet_pool.sample(1, random_state=SEED).iloc[0]))

    eb_pool = manifest[manifest["label"] == "eb"]
    if len(eb_pool):
        targets.append(("eb", eb_pool.sample(1, random_state=SEED).iloc[0]))

    blend_pool = manifest[manifest["label"] == "blend"]
    if len(blend_pool):
        targets.append(("blend", blend_pool.sample(1, random_state=SEED).iloc[0]))

    return targets


def phase_fold_smoke_test(manifest):
    print("\n=== Phase-fold smoke test (3 PNGs: planet / eb-stellar-eclipse / blend-centroid-offset) ===")
    targets = pick_smoke_targets(manifest)
    if not targets:
        print("  no eligible targets found (planet/eb/blend)")
        return []

    saved = []
    for label, row in targets:
        kic_id = row["tic_id"]
        npz_path = PROC_DIR / label / f"{kic_id}.npz"
        if not npz_path.exists():
            print(f"  {label} KIC {kic_id}: .npz not found, skipping (not yet downloaded/preprocessed)")
            continue
        d = load_npz(npz_path)
        period = row.get("period_days")
        if period is None or not np.isfinite(period) or period <= 0:
            print(f"  {label} KIC {kic_id}: no valid period, skipping")
            continue

        flux = d["flux"]
        epoch = row.get("epoch_btjd")  # actually BKJD for Kepler; see README
        phase = phase_fold(d["time"], period, epoch)

        order = np.argsort(phase)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(phase[order], flux[order], s=2, alpha=0.5)
        ax.set_xlabel("phase")
        ax.set_ylabel("flux")
        note = {"planet": "expect U-shaped dip",
                "eb": "expect stellar-eclipse (on-target) dip, possibly V-shaped",
                "blend": "expect a shallow, possibly diluted/off-shape dip (real centroid-offset FP)"}[label]
        ax.set_title(f"{label} KIC {kic_id} folded at P={period:.4f}d\n{note}")
        out_path = PROC_DIR / f"smoke_{label}_{kic_id}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")
        saved.append(out_path)
    return saved


# --------------------------------------------------------------- report --
def final_report(manifest, coverage_counts):
    print("\n=== Final report ===")
    total_have, total_want = 0, 0
    total_size = 0
    for label, (have, want) in coverage_counts.items():
        total_have += have
        total_want += want
        label_dir = PROC_DIR / label
        if label_dir.exists():
            total_size += sum(p.stat().st_size for p in label_dir.glob("*.npz"))

    print("class counts (present/manifest):")
    for label, (have, want) in coverage_counts.items():
        print(f"  {label:10s} {have}/{want}")
    print(f"total size on disk: {total_size / 1e6:.1f} MB")
    fail_rate = 1 - (total_have / total_want) if total_want else 0
    print(f"failure rate: {fail_rate:.1%}")
    if total_want and fail_rate > 0.15:
        print("  WARNING: >15% missing -- Kepler MAST gaps should be rarer than TESS's; investigate.")


def main():
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False, na_values=[""])
    coverage_counts = check_coverage(manifest)
    spot_check(manifest)
    phase_fold_smoke_test(manifest)
    final_report(manifest, coverage_counts)


if __name__ == "__main__":
    main()
