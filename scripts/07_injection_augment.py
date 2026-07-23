#!/usr/bin/env python
"""Inject real/synthetic transit and eclipse signals into real TESS noise
curves (Olmschenk et al. / Planet Hunters TESS trick: sector systematics
and gaps come for free because the noise is real). See injection-
augmentation addendum, Phases B1-B3.

Donor noise curves are `data/processed/null/*.npz` (real TESS light curves
with no known signal). The TESS download/preprocess job may still be
writing into data/processed/ concurrently, so donor files are enumerated
once at startup and any unreadable/zero-size file is skipped and logged,
never crashing the run. This script only reads from data/processed/ and
never writes there -- all outputs go to data/augmented/.
"""
import argparse
import sys
from pathlib import Path

import batman
import numpy as np
import pandas as pd
import wotan
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DONOR_NULL_DIR = ROOT / "data" / "processed" / "null"
DONOR_EB_DIR = ROOT / "data" / "processed" / "eb"
OUT_DIR = ROOT / "data" / "augmented"

# Same detrending recipe as 03_preprocess.py -- if this drifts, the augmented
# and real distributions differ on preprocessing artifacts instead of
# astrophysics, and a classifier "wins" by detecting the fingerprint.
WINDOW_LENGTH_DAYS = 0.5

G = 6.674e-11          # m^3 kg^-1 s^-2
RHO_SUN = 1408.0        # kg/m^3, mean solar density
SECONDS_PER_DAY = 86400.0


# --------------------------------------------------------------- donors --
def safe_load_npz(path):
    try:
        if path.stat().st_size == 0:
            return None
        with np.load(path, allow_pickle=True) as d:
            return {k: d[k] for k in d.files}
    except Exception as e:
        print(f"  WARNING: skipping unreadable donor {path}: {e}")
        return None


def enumerate_donors():
    null_files = sorted(DONOR_NULL_DIR.glob("*.npz")) if DONOR_NULL_DIR.exists() else []
    eb_files = sorted(DONOR_EB_DIR.glob("*.npz")) if DONOR_EB_DIR.exists() else []
    null_donors = []
    for f in null_files:
        data = safe_load_npz(f)
        if data is not None:
            null_donors.append(data)
    eb_donors = []
    for f in eb_files:
        data = safe_load_npz(f)
        if data is not None:
            eb_donors.append(data)
    return null_donors, eb_donors


# ------------------------------------------------------------ transit model --
def a_over_rs_from_density(period_days, rho_star_solar):
    """a/R* via Kepler's third law + mean stellar density (Winn 2010 eq. 5)."""
    period_s = period_days * SECONDS_PER_DAY
    return (G * rho_star_solar * RHO_SUN * period_s ** 2 / (3 * np.pi)) ** (1 / 3)


def transit_duration_days(period_days, a_over_rs, rp, b):
    arg = (1.0 / a_over_rs) * np.sqrt((1 + rp) ** 2 - b ** 2)
    arg = np.clip(arg, -1.0, 1.0)
    return (period_days / np.pi) * np.arcsin(arg)


def count_full_transits(time, t0, period, duration):
    """How many expected transit epochs have >=3 real cadences within
    duration/2 of center -- a proxy for "a full transit lands in valid data".
    """
    if not np.isfinite(duration) or duration <= 0 or not np.isfinite(period) or period <= 0:
        return 0
    tmin, tmax = time.min(), time.max()
    n_min = int(np.floor((tmin - t0) / period)) - 1
    n_max = int(np.ceil((tmax - t0) / period)) + 1
    count = 0
    for n in range(n_min, n_max + 1):
        center = t0 + n * period
        if (np.abs(time - center) < duration / 2).sum() >= 3:
            count += 1
    return count


def batman_flux(time, t0, period, rp, a_over_rs, inc_deg, u):
    p = batman.TransitParams()
    p.t0 = t0
    p.per = period
    p.rp = rp
    p.a = a_over_rs
    p.inc = inc_deg
    p.ecc = 0.0
    p.w = 90.0
    p.u = list(u)
    p.limb_dark = "quadratic"
    model = batman.TransitModel(p, time)
    return model.light_curve(p)


# --------------------------------------------------------- planet injection --
def sample_planet_params(rng, tmin, tmax):
    period = 10 ** rng.uniform(np.log10(0.5), np.log10(15))
    depth = 10 ** rng.uniform(np.log10(200e-6), np.log10(0.02))
    rp = np.sqrt(depth)
    rho_star = rng.uniform(0.2, 3.0)
    a_over_rs = a_over_rs_from_density(period, rho_star)
    b = rng.uniform(0.0, 0.9)
    inc_deg = np.degrees(np.arccos(min(b / a_over_rs, 0.999)))
    t0 = rng.uniform(tmin, tmax)
    u = [rng.uniform(0.2, 0.5), rng.uniform(0.1, 0.3)]
    duration = transit_duration_days(period, a_over_rs, rp, b)
    return dict(period_days=period, depth=depth, rp=rp, rho_star_solar=rho_star,
                a_over_rs=a_over_rs, b=b, inc_deg=inc_deg, t0=t0, u=u,
                duration_days=duration)


def generate_planet_signal(time, rng, max_attempts=50):
    for _ in range(max_attempts):
        p = sample_planet_params(rng, time.min(), time.max())
        n_transits = count_full_transits(time, p["t0"], p["period_days"], p["duration_days"])
        if n_transits < 2:
            continue
        flux = batman_flux(time, p["t0"], p["period_days"], p["rp"], p["a_over_rs"], p["inc_deg"], p["u"])
        meta = {
            "kind": "planet", "period_days": p["period_days"], "t0": p["t0"],
            "depth": p["depth"], "rp": p["rp"], "rho_star_solar": p["rho_star_solar"],
            "a_over_rs": p["a_over_rs"], "b": p["b"], "inc_deg": p["inc_deg"],
            "u1": p["u"][0], "u2": p["u"][1], "n_transits_landed": n_transits,
        }
        return flux, meta
    return None, None


# ------------------------------------------------------------- eb injection --
def sample_eb_batman_params(rng, tmin, tmax):
    period = 10 ** rng.uniform(np.log10(0.5), np.log10(15))
    rp = np.sqrt(rng.uniform(0.05, 0.5))  # large radius ratio -> deep primary
    rho_star = rng.uniform(0.2, 3.0)
    a_over_rs = a_over_rs_from_density(period, rho_star)
    b = rng.uniform(0.0, 0.7)
    inc_deg = np.degrees(np.arccos(min(b / a_over_rs, 0.999)))
    t0 = rng.uniform(tmin, tmax)
    u = [rng.uniform(0.2, 0.5), rng.uniform(0.1, 0.3)]
    secondary_depth_ratio = rng.uniform(0.1, 0.8)
    duration = transit_duration_days(period, a_over_rs, rp, b)
    return dict(period_days=period, rp=rp, rho_star_solar=rho_star, a_over_rs=a_over_rs,
                b=b, inc_deg=inc_deg, t0=t0, u=u,
                secondary_depth_ratio=secondary_depth_ratio, duration_days=duration)


def generate_eb_batman_signal(time, rng, max_attempts=50):
    for _ in range(max_attempts):
        p = sample_eb_batman_params(rng, time.min(), time.max())
        n_transits = count_full_transits(time, p["t0"], p["period_days"], p["duration_days"])
        if n_transits < 2:
            continue
        primary = batman_flux(time, p["t0"], p["period_days"], p["rp"], p["a_over_rs"], p["inc_deg"], p["u"])
        rp_secondary = p["rp"] * np.sqrt(p["secondary_depth_ratio"])
        secondary = batman_flux(
            time, p["t0"] + p["period_days"] / 2, p["period_days"], rp_secondary,
            p["a_over_rs"], p["inc_deg"], p["u"],
        )
        flux = primary + secondary - 1.0  # both baseline at 1; sum double-counts it
        meta = {
            "kind": "eb_batman", "period_days": p["period_days"], "t0": p["t0"],
            "rp_primary": p["rp"], "rp_secondary": rp_secondary,
            "secondary_depth_ratio": p["secondary_depth_ratio"],
            "rho_star_solar": p["rho_star_solar"], "a_over_rs": p["a_over_rs"],
            "b": p["b"], "inc_deg": p["inc_deg"], "u1": p["u"][0], "u2": p["u"][1],
            "n_transits_landed": n_transits,
        }
        return flux, meta
    return None, None


def generate_eb_resampled_signal(time, rng, eb_donors, max_attempts=20):
    if not eb_donors:
        return None, None
    for _ in range(max_attempts):
        src = eb_donors[rng.integers(len(eb_donors))]
        period = float(src.get("period_days", np.nan))
        if not np.isfinite(period) or period <= 0:
            continue
        t0 = rng.uniform(time.min(), time.max())
        # a real EB's eclipses are much narrower than its period; ~5% of P is
        # a reasonable stand-in duration purely for the landed-transits check.
        n_transits = count_full_transits(time, t0, period, duration=period * 0.05)
        if n_transits < 2:
            continue

        src_time = np.asarray(src["time"], dtype=np.float64)
        src_flux = np.asarray(src["flux"], dtype=np.float64)
        epoch_src = float(src.get("epoch_btjd", np.nan))
        if not np.isfinite(epoch_src):
            epoch_src = float(src_time[0])

        phase_src = ((src_time - epoch_src) / period) % 1.0
        order = np.argsort(phase_src)
        phase_sorted = phase_src[order]
        flux_sorted = src_flux[order] / np.nanmedian(src_flux)
        # wrap so np.interp handles the phase-0/1 boundary correctly
        phase_wrap = np.concatenate([phase_sorted - 1.0, phase_sorted, phase_sorted + 1.0])
        flux_wrap = np.concatenate([flux_sorted, flux_sorted, flux_sorted])

        phase_target = ((time - t0) / period) % 1.0
        flux = np.interp(phase_target, phase_wrap, flux_wrap)
        meta = {
            "kind": "eb_resampled",
            "source_tic_id": int(src["tic_id"]) if "tic_id" in src else None,
            "period_days": period, "t0": t0, "n_transits_landed": n_transits,
        }
        return flux, meta
    return None, None


def generate_eb_signal(time, rng, eb_donors, max_attempts=50):
    use_resampled_first = rng.random() < 0.5
    order = ([generate_eb_resampled_signal, generate_eb_batman_signal] if use_resampled_first
              else [generate_eb_batman_signal, generate_eb_resampled_signal])
    for gen in order:
        flux, meta = (gen(time, rng, eb_donors, max_attempts) if gen is generate_eb_resampled_signal
                      else gen(time, rng, max_attempts))
        if flux is not None:
            return flux, meta
    return None, None


# --------------------------------------------------------- blend injection --
def generate_blend_signal(time, rng, eb_donors, max_attempts=50):
    if rng.random() < 0.5:
        base_flux, base_meta = generate_planet_signal(time, rng, max_attempts)
    else:
        base_flux, base_meta = generate_eb_signal(time, rng, eb_donors, max_attempts)
    if base_flux is None:
        return None, None
    f = rng.uniform(0.1, 0.6)
    diluted = 1.0 + f * (base_flux - 1.0)
    meta = {"kind": "blend", "crowdsap": f, "base_signal": base_meta}
    return diluted, meta


def _estimate_duration_days(meta):
    """Best-available transit duration for masking the redetrend fit, given
    whichever signal generator produced ``meta``. batman-generated signals
    (planet, eb_batman) carry the physical params to compute it exactly;
    eb_resampled has no such model, so this falls back to the same
    period*0.05 stand-in count_full_transits already uses for that
    generator. Returns None if not even a period is available.
    """
    period = meta.get("period_days")
    if period is None or not np.isfinite(period) or period <= 0:
        return None
    kind = meta.get("kind")
    rp = meta.get("rp") or meta.get("rp_primary")
    a_over_rs = meta.get("a_over_rs")
    b = meta.get("b")
    if kind in ("planet", "eb_batman") and rp is not None and a_over_rs is not None and b is not None:
        return transit_duration_days(period, a_over_rs, rp, b)
    return period * 0.05


def _injection_mask(time, meta):
    """In-transit mask for the redetrend fit, or None if the ephemeris
    isn't known. See in_transit_mask() in 03_preprocess.py -- this is the
    same fix, needed here too since inject_and_redetrend() re-runs the
    identical unmasked wotan.flatten() call on the freshly-injected signal."""
    if meta is None:
        return None
    base = meta.get("base_signal") if meta.get("kind") == "blend" else meta
    if not base:
        return None
    period = base.get("period_days")
    t0 = base.get("t0")
    duration = _estimate_duration_days(base)
    if period is None or t0 is None or duration is None:
        return None
    if not (np.isfinite(period) and np.isfinite(t0) and np.isfinite(duration) and period > 0 and duration > 0):
        return None
    half_dur = duration / 2.0
    phase = np.mod(time - t0 + period / 2.0, period) - period / 2.0
    return np.abs(phase) < half_dur


# --------------------------------------------------------- inject + redetrend --
def inject_and_redetrend(donor, signal, meta=None):
    time = np.asarray(donor["time"], dtype=np.float64)
    raw = donor["flux_raw"] if "flux_raw" in donor else donor["flux"]
    raw = np.asarray(raw, dtype=np.float64)
    flux_err = np.asarray(donor["flux_err"], dtype=np.float64)

    injected_raw = raw * signal
    mask = _injection_mask(time, meta)
    flattened, trend = wotan.flatten(
        time, injected_raw, method="biweight", window_length=WINDOW_LENGTH_DAYS,
        return_trend=True, mask=mask,
    )
    flux_err_detrended = flux_err / trend
    median_final = np.nanmedian(flattened)
    flux_norm = flattened / median_final
    flux_err_norm = flux_err_detrended / median_final
    return time, flux_norm, flux_err_norm


def build_one(label, donors_null, donors_eb, rng):
    donor = donors_null[rng.integers(len(donors_null))]
    time = np.asarray(donor["time"], dtype=np.float64)

    if label == "planet":
        signal, meta = generate_planet_signal(time, rng)
    elif label == "eb":
        signal, meta = generate_eb_signal(time, rng, donors_eb)
    elif label == "blend":
        signal, meta = generate_blend_signal(time, rng, donors_eb)
    else:
        raise ValueError(f"unknown class {label}")

    if signal is None:
        return None

    time_out, flux_out, flux_err_out = inject_and_redetrend(donor, signal, meta)

    if label == "blend":
        crowdsap = meta["crowdsap"]
        crowdsap_missing = False
        base = meta.get("base_signal") or {}
        period_days = base.get("period_days", np.nan)
        epoch = base.get("t0", np.nan)
    else:
        crowdsap = donor.get("crowdsap", 1.0)
        crowdsap_missing = donor.get("crowdsap_missing", True)
        period_days = meta.get("period_days", np.nan)
        epoch = meta.get("t0", np.nan)

    return {
        "time": time_out,
        "flux": flux_out,
        "flux_err": flux_err_out,
        "tic_id": donor.get("tic_id"),
        "label": label,
        "sector": donor.get("sector", -1),
        "period_days": period_days,
        "epoch_btjd": epoch,
        "crowdsap": crowdsap,
        "crowdsap_missing": crowdsap_missing,
        "augmented": True,
        "donor_tic_id": donor.get("tic_id"),
        "injection_params": meta,
    }


# --------------------------------------------------------------------- verify --
def verify(classes):
    import random as pyrandom
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n=== Verify: spot checks (5/class) + phase-fold PNGs (3) + counts ===")
    rng = pyrandom.Random(42)
    total_size = 0
    for label in classes:
        out_dir = OUT_DIR / label
        files = sorted(out_dir.glob("*.npz")) if out_dir.exists() else []
        if not files:
            print(f"  {label:8s} no files")
            continue
        size = sum(f.stat().st_size for f in files)
        total_size += size
        print(f"  {label:8s} {len(files)} files, {size / 1e6:.1f} MB")

        sample = pyrandom.Random(42).sample(files, min(5, len(files)))
        issues = []
        for f in sample:
            with np.load(f, allow_pickle=True) as d:
                flux = d["flux"]
                if np.isnan(flux).any():
                    issues.append(f"{f.name}: NaNs present")
                med = np.nanmedian(flux)
                if not np.isclose(med, 1.0, atol=0.05):
                    issues.append(f"{f.name}: median {med:.3f} != 1.0")
        if issues:
            print("    ISSUES: " + "; ".join(issues))
        else:
            print(f"    spot-checked {len(sample)} files, no issues")

        example = rng.choice(files)
        with np.load(example, allow_pickle=True) as d:
            time, flux = d["time"], d["flux"]
            period, epoch = d["period_days"], d["epoch_btjd"]
            if not (np.isfinite(period) and period > 0 and np.isfinite(epoch)):
                print(f"    {example.name}: no valid injected period for phase-fold, skipping PNG")
                continue
            phase = ((time - float(epoch)) / float(period)) % 1.0
            order = np.argsort(phase)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(phase[order], flux[order], s=2, alpha=0.5)
            ax.set_xlabel("phase")
            ax.set_ylabel("flux")
            shape_note = {"planet": "expect U-dip", "eb": "expect primary+secondary",
                          "blend": "expect a shallow diluted dip"}[label]
            ax.set_title(f"augmented {label} {example.stem} folded at P={float(period):.4f}d\n{shape_note}")
            png_path = OUT_DIR / f"smoke_{label}_{example.stem}.png"
            fig.savefig(png_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"    saved {png_path}")

    print(f"\ntotal augmented size on disk: {total_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--classes", default="planet,eb,blend")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    rng = np.random.default_rng(args.seed)

    print("Enumerating donor files (safe against a concurrently-running TESS preprocess job)...")
    donors_null, donors_eb = enumerate_donors()
    print(f"  null donors (noise): {len(donors_null)}   eb donors (for resampling): {len(donors_eb)}")
    if not donors_null:
        print("ERROR: no usable null donor files in data/processed/null/. Nothing to inject into.",
              file=sys.stderr)
        sys.exit(1)
    if "eb" in classes or "blend" in classes:
        if not donors_eb:
            print("  WARNING: no eb donors found; eb generation will fall back to pure-batman EBs only.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for label in classes:
        out_dir = OUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(out_dir.glob("*.npz")))
        if existing >= args.n_per_class:
            print(f"{label}: already have {existing} >= --n-per-class {args.n_per_class}, skipping")
            continue

        n_to_make = args.n_per_class - existing
        made, failed = 0, 0
        pbar = tqdm(range(n_to_make), desc=f"injecting {label}")
        idx = existing
        for _ in pbar:
            result = build_one(label, donors_null, donors_eb, rng)
            if result is None:
                failed += 1
                pbar.set_postfix(made=made, failed=failed)
                continue
            out_path = out_dir / f"aug_{idx:06d}.npz"
            np.savez(out_path, **result)
            idx += 1
            made += 1
            pbar.set_postfix(made=made, failed=failed)
        print(f"{label}: made={made} failed={failed} (gave up after max_attempts) -> {out_dir}")

    verify(classes)


if __name__ == "__main__":
    main()
