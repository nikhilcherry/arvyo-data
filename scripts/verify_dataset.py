#!/usr/bin/env python
"""Sanity checks + summary stats over data/processed/. See Phase 4 of the handout."""
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "manifest.csv"
PROC_DIR = ROOT / "data" / "processed"
SUSPECTS_PATH = ROOT / "data" / "null_suspects.csv"

SEED = 42
SPOT_CHECK_N = 5
NULL_CHECK_N = 10
NULL_FLAG_RATIO = 3.0


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
            if len(flux) < 1000:
                issues.append(f"only {len(flux)} cadences (<1000)")
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

    planet_pool = manifest[(manifest["label"] == "planet")
                            & (manifest["disposition"].astype(str).str.upper() == "CP")]
    if len(planet_pool):
        targets.append(("planet", planet_pool.sample(1, random_state=SEED).iloc[0]))

    eb_pool = manifest[(manifest["label"] == "eb") & (manifest["period_days"] < 5)]
    if len(eb_pool):
        targets.append(("eb", eb_pool.sample(1, random_state=SEED).iloc[0]))

    spot_pool = manifest[manifest["label"] == "starspot"]
    if len(spot_pool):
        targets.append(("starspot", spot_pool.sample(1, random_state=SEED).iloc[0]))

    return targets


def phase_fold_smoke_test(manifest):
    print("\n=== Phase-fold smoke test (3 PNGs) ===")
    targets = pick_smoke_targets(manifest)
    if not targets:
        print("  no eligible targets found (planet/CP, eb/period<5d, starspot)")
        return []

    saved = []
    for label, row in targets:
        tic_id = row["tic_id"]
        npz_path = PROC_DIR / label / f"{tic_id}.npz"
        if not npz_path.exists():
            print(f"  {label} TIC {tic_id}: .npz not found, skipping")
            continue
        d = load_npz(npz_path)
        period = row.get("period_days")
        if period is None or not np.isfinite(period) or period <= 0:
            print(f"  {label} TIC {tic_id}: no valid period, skipping")
            continue

        flux = d["flux_raw"] if label == "starspot" and "flux_raw" in d else d["flux"]
        epoch = row.get("epoch_btjd")
        phase = phase_fold(d["time"], period, epoch)

        order = np.argsort(phase)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(phase[order], flux[order], s=2, alpha=0.5)
        ax.set_xlabel("phase")
        ax.set_ylabel("flux_raw" if "flux_raw" in d and label == "starspot" else "flux")
        ax.set_title(f"{label} TIC {tic_id} folded at P={period:.4f}d")
        out_path = PROC_DIR / f"smoke_{label}_{tic_id}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_path}")
        saved.append(out_path)
    return saved


# --------------------------------------------------------- null purity --
def null_purity_check(manifest):
    print(f"\n=== Null purity check ({NULL_CHECK_N} random targets) ===")
    from astropy.timeseries import LombScargle

    null_dir = PROC_DIR / "null"
    files = sorted(null_dir.glob("*.npz")) if null_dir.exists() else []
    if not files:
        print("  no null .npz files found")
        return

    rng = random.Random(SEED)
    sample = rng.sample(files, min(NULL_CHECK_N, len(files)))

    suspects = []
    for f in sample:
        d = load_npz(f)
        time, flux = d["time"], d["flux"]
        tic_id = int(d["tic_id"])
        try:
            freq, power = LombScargle(time, flux).autopower(
                minimum_frequency=1 / 15.0, maximum_frequency=1 / 0.5,
            )
            best_period = 1.0 / freq[np.argmax(power)]
            phase = phase_fold(time, best_period)
            bins = np.linspace(0, 1, 21)
            idx = np.digitize(phase, bins)
            binned_means = np.array([
                flux[idx == i].mean() for i in range(1, 21) if np.any(idx == i)
            ])
            std_binned = np.std(binned_means)
            point_noise = np.std(np.diff(flux)) / np.sqrt(2)
            ratio = std_binned / point_noise if point_noise > 0 else np.inf
        except Exception as e:
            print(f"  TIC {tic_id}: periodogram failed ({e}), skipping")
            continue

        flagged = ratio > NULL_FLAG_RATIO
        if flagged:
            suspects.append({
                "tic_id": tic_id, "best_period_days": best_period,
                "std_binned_over_noise": ratio,
            })
        print(f"  TIC {tic_id}: best_period={best_period:.2f}d ratio={ratio:.2f}"
              f"{'  <-- FLAGGED' if flagged else ''}")

    if suspects:
        pd.DataFrame(suspects).to_csv(SUSPECTS_PATH, index=False)
        print(f"  {len(suspects)} suspects -> {SUSPECTS_PATH}")
    else:
        print("  no suspects flagged")


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


def main():
    # keep_default_na=False: pandas' default NA sentinels include the literal
    # string "null", which is one of our label values (the quiet-star class).
    manifest = pd.read_csv(MANIFEST_PATH, keep_default_na=False, na_values=[""])
    coverage_counts = check_coverage(manifest)
    spot_check(manifest)
    phase_fold_smoke_test(manifest)
    null_purity_check(manifest)
    final_report(manifest, coverage_counts)


if __name__ == "__main__":
    main()
