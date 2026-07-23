# arvyo-data

Dataset repository for team Arvyo's BAH2026 exoplanet transit-detection
pipeline. Builds a labeled set of TESS light curves for a 4-class classifier
(planet / eclipsing binary / blend / unknown), plus supporting `starspot` and
`null` classes used to validate the blend/novelty forward models against
real stellar variability.

## Label classes and sources

| label      | source                                                                 |
|------------|-------------------------------------------------------------------------|
| `planet`   | ExoFOP TOI table, `TFOPWG Disposition` in {`CP`, `KP`}                  |
| `unknown`  | ExoFOP TOI table, `TFOPWG Disposition` = `FP` (kept as a grab-bag; some FPs are blends and may be relabeled later) |
| `eb`       | TESS-EB catalog (Prša et al. 2022), villanova.edu / MAST HLSP           |
| `starspot` | A TESS rotation-period catalog (e.g. Kounkel et al. 2022) fetched via Vizier, `Prot` in [0.5, 15] days |
| `null`     | Quiet-by-exclusion: TIC stars (Tmag 8-11) not present in any of the above catalogs |
| `blend`    | Not populated here — generated synthetically downstream |

`manifest_candidates.csv` holds TOI rows with disposition `PC` or blank,
excluded from the training manifest, for later inference/demo use.

## Rebuilding from scratch

```
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/01_build_manifest.py          # -> manifest.csv, manifest_full.csv, manifest_candidates.csv
python scripts/02_download_lightcurves.py --limit 10   # verify a small batch first
python scripts/02_download_lightcurves.py               # full download (1-3 hours)
python scripts/03_preprocess.py
python scripts/verify_dataset.py
```

All three pipeline scripts are resumable and safe to re-run after a crash.
To scale the dataset up, re-run `01_build_manifest.py` with larger `--n-*`
flags, then `02` and `03` again — only new targets are downloaded/processed.

## Data hosting

`data/` is git-ignored (light curves total several GB). The bulk dataset is
hosted on Kaggle Datasets: `<placeholder — Kaggle link TBD>`.

## Samples

`data/samples/` is the one exception to the git-ignore above, and the only
data this repo actually commits: one small, real, contract-valid `.npz` per
label (`planet`, `eb`, `blend`, `starspot`, `null`), each ≤500KB, for
smoke-testing arvyo-pipeline against real data without pulling the full
corpus. `blend` comes from the real Kepler DR25 centroid-offset false
positives in `data/kepler/processed/blend/` (see "Kepler DR25 transfer set"
below) — the TESS `blend` class is synthetic-only, so it's excluded as a
source here.

Regenerate with:

```
python scripts/export_samples.py --seed 42
```

Each label's sample is picked uniformly at random (seeded, reproducible)
from that label's contract-valid files already under the 500KB budget, so
the committed sample isn't cherry-picked to look unusually clean. `data/samples/PROVENANCE.md`
records exactly which source file was used per label, and is regenerated
alongside the samples — do not hand-edit either. Full training/eval runs
should still use the manifest and the full corpus (Kaggle or a local
rebuild), not `data/samples/`.

The `blend` row is the one exception to "picked uniformly at random": it's
pinned via `scripts/export_samples.py --force blend=<path>` to a specific,
pre-registered target (KIC 4281068 / KOI K07689.01) rather than a random
draw, because an earlier random pick (KIC 6974867) turned out to sit right
at the detection floor — TLS recovered SDE ≈ 5.0 against arvyo-pipeline's
7.0 gate, so the worker's period-search stage short-circuited to
`no_period` and the fitr model-comparison stage never ran on it. The
replacement was chosen with `scripts/08_select_blend_sample.py`, which
filters the DR25 KOI cumulative table for `FALSE POSITIVE`s with the
centroid-offset flag (`koi_fpflag_co == 1`), an on-target-shaped signal
(`koi_fpflag_nt == 0`), a short period (0.5-5.0 d, for many events per
quarter), catalog depth ≥500 ppm, and catalog `koi_model_snr` ≥20, then
ranks survivors by SNR descending; the results are recorded in
`data/catalogs/blend_candidates.csv`. On the new target, TLS recovers
SDE ≈ 18.9 (period within 0.02% of the catalog value) and fitr's 4-model
comparison runs to a `clear` verdict — though it picks `eb`, not `blend`,
as the best-fit model (see PROVENANCE and the task report for the full
before/after numbers). `koi_fpflag_co` reflects a Robovetter centroid-shift
determination that isn't necessarily recoverable from light-curve shape
alone, so this mismatch is expected and reported as-is rather than treated
as a reason to pick a different candidate.

---

## Kepler DR25 transfer set

A second, independent labeled set built from Kepler (not TESS) to use as a
pretraining/transfer source and — critically — as a source of **real** blend
(background eclipsing binary) examples, since the TESS `blend` class here is
synthetic-only.

### Sources and label mapping

All three tables come from the NASA Exoplanet Archive TAP service
(`exoplanetarchive.ipac.caltech.edu/TAP/sync`):

- `q1_q17_dr25_tce` — the Q1-Q17 DR25 TCE list (the Robovetter's input
  population; fetched and cached for reference, not joined into the
  manifest since the cumulative KOI table already carries the dispositions
  and FP flags we need).
- `cumulative` — the Cumulative KOI table: dispositions (`koi_disposition`)
  and Robovetter FP flags (`koi_fpflag_nt/ss/co/ec`).
- `fpwg` (Certified False Positive table) — **not available** on the current
  TAP service (confirmed absent from `TAP_SCHEMA.tables`; it appears to have
  been retired/merged into the cumulative dispositions). The build script
  logs this, prints manual-download instructions, and continues using
  flag-derived labels only. If a `certified_fp.csv` is later placed at
  `data/kepler/catalogs/certified_fp.csv`, re-running the script will pick
  it up and cross-check against it automatically.

Labels (mirrors the TESS taxonomy so both manifests can be mixed in one
training loop), per Thompson et al. 2018 (the Kepler DR25 Robovetter paper):

| condition                                              | label     |
|---------------------------------------------------------|-----------|
| `koi_disposition == CONFIRMED`                           | `planet`  |
| `koi_fpflag_co == 1`                                     | `blend`   |
| `koi_fpflag_ss == 1` and `koi_fpflag_co == 0`             | `eb`      |
| `koi_fpflag_nt == 1` and both above == 0                  | `unknown` |
| `koi_disposition == CANDIDATE`                            | excluded → `kepler_manifest_candidates.csv` |

Flags can co-occur; priority is `co` (blend) > `ss` (eb) > `nt` (unknown) —
a centroid offset means the signal isn't on-target regardless of its shape.

**Why blend comes from centroid-offset flags:** `koi_fpflag_co` marks TCEs
where the Robovetter detected the transit signal centroid displaced from the
target star — i.e. a real background/nearby eclipsing binary diluted into
the target's aperture. This is the one class TESS alone can't supply real
examples of (our TESS `blend` class is purely synthetic), so these
Kepler-vetted centroid-offset FPs are the intended ground truth to validate
the synthetic blend generator against.

### ID namespace

The `tic_id` column holds the **KIC ID** for Kepler rows (not renamed to
`star_id` — doing so would require touching the shared manifest-schema
convention while the TESS pipeline may be actively reading/writing
`manifest.csv`). A `mission` column (`kepler` vs `tess`) disambiguates the
namespace; never join the two manifests on `tic_id` alone. `epoch_btjd` for
Kepler rows is actually **BKJD** (`koi_time0bk`, BJD − 2454833), not BTJD —
same concept, different zero-point offset.

A KIC can host multiple KOIs (multi-planet systems): the manifest keeps one
row per KOI, but the download/preprocess scripts dedupe by KIC id since
`.npz` files are keyed by star, not by KOI.

### Pipeline

```
python scripts/04_build_kepler_manifest.py   # -> kepler_manifest.csv, _full, _candidates
python scripts/05_download_kepler.py --limit 10   # verify a small batch first
python scripts/05_download_kepler.py               # full download (~4,100 quarters)
python scripts/06_preprocess_kepler.py
python scripts/verify_kepler.py
```

Kepler long cadence (30-min) is used — DR25 vetting was done on it, and it
keeps download size down — with one quarter (the first available)
downloaded per target. `.npz` schema matches the TESS one exactly (`time`,
`flux`, `flux_err`, `tic_id`, `label`, `sector`, `period_days`,
`epoch_btjd`, `crowdsap`, `crowdsap_missing`) plus `mission="kepler"`. The
too-few-cadences floor is lowered to 500 (from TESS's stricter floor) since
Kepler LC quarters are much shorter than a TESS 2-min sector.

### Alternative pretraining source (not integrated)

The Shallue & Vanderburg / AstroNet preprocessed global/local-view
TFRecords (github.com/google-research/exoplanet-ml) cover 15,737 labeled
Kepler TCEs and are a possible shortcut to a larger pretraining set. They
are **not** integrated into this repo — different preprocessing lineage —
but are worth knowing about as an alternative source.

### DR25 pixel-level injection products (documentation only)

The Kepler DR25 pixel-level transit injection products (NASA Exoplanet
Archive bulk downloads) are the planned ground truth for SBI
posterior-calibration tests (inject with known params → fit → coverage
check). They are **not** downloaded in this repo — they're huge and belong
to the modeling repo — this is just a pointer for later:
https://exoplanetarchive.ipac.caltech.edu/docs/PurposeOfKOITable.html
(see the DR25 injected-transit / pixel-level injection bulk-download pages).

---

## Injection augmentation

`scripts/07_injection_augment.py` multiplies the training set by injecting
real/synthetic transit and eclipse signals into **real TESS noise curves**
(donors from `data/processed/null/*.npz`). This is the Olmschenk et al. /
Planet Hunters TESS trick: sector systematics and gaps come for free
because the noise is real, only the signal is injected.

Three recipes, each producing `.npz` files with the same schema as
`03_preprocess.py` plus `augmented=True` and the full injected parameter
vector (free ground truth for later injection-recovery tests of TLS/SBI):

- **planet** — a `batman` quadratic-limb-darkening transit (period
  0.5–15 d log-uniform, depth 200 ppm–2% log-uniform, duration consistent
  with a plausible stellar density, random epoch), multiplied into a null
  donor. Rejects combos with < 2 full transits in valid cadences.
- **eb** — 50/50 either a `batman` eclipse with a large radius ratio plus a
  secondary eclipse (depth ratio 0.1–0.8 at phase 0.5), or a real
  phase-folded EB resampled from `data/processed/eb/` and multiplied into
  the donor (real-signal-into-real-noise).
- **blend** — either generator above, diluted by a crowding factor
  `f ∈ [0.1, 0.6]` (`flux = 1 + f*(signal-1)`), with `crowdsap = f` recorded
  — a synthetic blend class now built on real noise instead of pure
  synthetics.

Augmented curves are re-detrended with the exact same wotan settings as
`03_preprocess.py` (biweight, 0.5 d window, in-transit points masked out
of the fit using the injected ephemeris) so the augmented and real
distributions don't diverge on preprocessing artifacts.

```
python scripts/07_injection_augment.py --n-per-class 1000 --seed 42
```

Outputs go to `data/augmented/{planet,eb,blend}/`.

## Parallel-run note

This repo's TESS download/preprocess (`02_download_lightcurves.py`,
`03_preprocess.py`) may run concurrently with the Kepler and augmentation
work above. The Kepler scripts never touch `data/raw/`, `data/processed/`,
or `manifest.csv`; all Kepler outputs go under `data/kepler/`. The
augmentation script only *reads* from `data/processed/{null,eb}/` (never
writes there) and skips any donor file that's zero-size or fails to load,
to tolerate files the TESS job is still writing.
