# mcmc_uncertainties.py
# Purpose: posterior sampling with emcee to quantify age/mass/feh uncertainties
#          for reliable stars from stellar_ages_results.csv.
# Output:  Outputs/mcmc_results.csv  — 16th/50th/84th percentiles per star.
#
# Run from a terminal:  python3 mcmc_uncertainties.py
# (multiprocessing requires __main__ guard — VSCode play button won't work)

import os, time
import numpy as np
import pandas as pd
import emcee
from scipy.interpolate import RegularGridInterpolator
from multiprocessing import Pool

# ── Paths ─────────────────────────────────────────────────────────────────────
_ANALYSIS   = os.path.dirname(os.path.abspath(__file__))
_BASE       = os.path.dirname(_ANALYSIS)
RESULTS_CSV = os.path.join(_BASE, 'Outputs', 'stellar_ages_results.csv')
PHOT_CSV    = os.path.join(_ANALYSIS, 'ASTR502_Master_Photometry_List.csv')
TARGET_CSV  = os.path.join(_ANALYSIS, 'ASTR502_Mega_Target_List.csv')
MCMC_CSV    = os.path.join(_BASE, 'Outputs', 'mcmc_results.csv')

# ── Sampler settings ──────────────────────────────────────────────────────────
N_WALKERS  = 32     # must be > 2 * ndim (ndim=3)
N_BURN     = 300    # steps to discard as burn-in
N_STEPS    = 600    # production steps
N_STARS    = None   # set to int (e.g. 10) for a test run; None = all reliable
N_CORES    = 4      # parallel processes; set to 1 to disable multiprocessing

# ── Priors — [] matches pure-photometry batch fit ─────────────────────────────
ACTIVE_PRIORS = []

# ── Grid ──────────────────────────────────────────────────────────────────────
_data     = np.load(os.path.join(_BASE, 'Grids', 'Tera Grid.npz'))
_reserved = ['masses', 'logages', 'fehs', 'Teff', 'logg']
_grid     = {k: _data[k] for k in _data if k not in _reserved}
masses    = _data['masses']
logages   = _data['logages']
fehs_g    = _data['fehs']
_teff_g   = _data['Teff'] if 'Teff' in _data else None
_logg_g   = _data['logg'] if 'logg' in _data else None


def _fill_edge_nans(arr):
    out = arr.copy()
    for i_f in range(arr.shape[0]):
        for i_la in range(arr.shape[1]):
            row = out[i_f, i_la, :]
            if np.all(np.isnan(row)):
                continue
            valid = np.where(np.isfinite(row))[0]
            if valid[0] > 0:
                out[i_f, i_la, :valid[0]] = row[valid[0]]
            if valid[-1] < arr.shape[2] - 1:
                out[i_f, i_la, valid[-1]+1:] = row[valid[-1]]
    return out


_grid = {k: _fill_edge_nans(v) for k, v in _grid.items()}
if _teff_g is not None:
    _teff_g = _fill_edge_nans(_teff_g)
if _logg_g is not None:
    _logg_g = _fill_edge_nans(_logg_g)

# Module-level interpolators — must be at top level for multiprocessing to work
_interp = {
    band: RegularGridInterpolator(
        (fehs_g, logages, masses), _grid[band],
        bounds_error=False, fill_value=np.nan, method='linear',
    )
    for band in _grid
}
_teff_interp = RegularGridInterpolator(
    (fehs_g, logages, masses), _teff_g,
    bounds_error=False, fill_value=np.nan, method='linear',
) if _teff_g is not None else None
_logg_interp = RegularGridInterpolator(
    (fehs_g, logages, masses), _logg_g,
    bounds_error=False, fill_value=np.nan, method='linear',
) if _logg_g is not None else None

# ── Photometry setup ──────────────────────────────────────────────────────────
ACTIVE_BANDS = ['G', 'BP', 'RP', 'J', 'H', 'K', 'W1', 'W2', 'W3', 'W4']
BAND_COLS = {
    'G':  ('gaiaGmag',  'e_gaiaGmag',  0.01),
    'BP': ('gaiaBPmag', 'e_gaiaBPmag', 0.02),
    'RP': ('gaiaRPmag', 'e_gaiaRPmag', 0.02),
    'J':  ('Jmag',      'e_Jmag',      0.03),
    'H':  ('Hmag',      'e_Hmag',      0.03),
    'K':  ('Kmag',      'e_Kmag',      0.03),
    'W1': ('w1mag',     'e_w1mag',     0.03),
    'W2': ('w2mag',     'e_w2mag',     0.03),
    'W3': ('w3mag',     'e_w3mag',     0.03),
    'W4': ('w4mag',     'e_w4mag',     0.05),
}
FEH_TOLERANCE = 0.15

# ── Log-probability (must be importable at module level for multiprocessing) ──
def log_prob(theta, observed, errors, bands, feh_lo, feh_hi,
             obs_teff=None, obs_logg=None):
    mass, logage, feh = theta

    if not (0.08 <= mass <= 3.0 and
            logages[0] <= logage <= logages[-1] and
            feh_lo <= feh <= feh_hi):
        return -np.inf

    point = [[feh, logage, mass]]
    lp = 0.0
    n_valid = 0
    for band in bands:
        if band not in _interp:
            continue
        val = float(np.asarray(_interp[band](point)).item())
        if not np.isfinite(val):
            continue
        lp     -= 0.5 * ((observed[band] - val) / errors[band]) ** 2
        n_valid += 1
    if n_valid < 3:
        return -np.inf

    if obs_teff is not None and 'teff' in ACTIVE_PRIORS and _teff_interp is not None:
        mt = float(np.asarray(_teff_interp(point)).item())
        if np.isfinite(mt):
            lp -= 0.5 * ((obs_teff - mt) / 100.0) ** 2

    if obs_logg is not None and 'logg' in ACTIVE_PRIORS and _logg_interp is not None:
        ml = float(np.asarray(_logg_interp(point)).item())
        if np.isfinite(ml):
            lp -= 0.5 * ((obs_logg - ml) / 0.1) ** 2

    return lp


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':

    results_df = pd.read_csv(RESULTS_CSV)
    reliable   = results_df[results_df['reliable'] == True].copy()
    reliable   = reliable[reliable['age_gyr'] >= 0.25].reset_index(drop=True)

    phot_df  = pd.read_csv(PHOT_CSV).drop_duplicates('hostname', keep='first')
    target_df = pd.read_csv(TARGET_CSV).drop_duplicates('hostname', keep='first')
    merged   = phot_df.merge(
        target_df[['hostname', 'bj_dist_pc', 'st_met', 'st_teff', 'st_logg']],
        on='hostname', how='inner',
    ).set_index('hostname')

    if N_STARS is not None:
        reliable = reliable.head(N_STARS)

    n_total = len(reliable)
    print(f"Grid: {len(fehs_g)} [Fe/H] × {len(logages)} logage × {len(masses)} mass")
    print(f"Active priors: {ACTIVE_PRIORS if ACTIVE_PRIORS else 'none (pure photometry)'}")
    print(f"\nRunning MCMC on {n_total} reliable stars")
    print(f"  {N_BURN} burn-in + {N_STEPS} production steps, {N_WALKERS} walkers, {N_CORES} cores")

    _t0     = time.time()
    records = []
    skipped = 0

    for i, fit_row in enumerate(reliable.itertuples()):
        hostname = fit_row.hostname

        if hostname not in merged.index:
            skipped += 1
            continue
        row = merged.loc[hostname]

        dist_pc = row['bj_dist_pc']
        st_met  = row['st_met']
        st_teff = row['st_teff']
        st_logg = row['st_logg']
        if not (np.isfinite(dist_pc) and np.isfinite(st_met)):
            skipped += 1
            continue

        mu = 5 * np.log10(dist_pc / 10)
        observed, errors = {}, {}
        for band, (mag_col, err_col, min_err) in BAND_COLS.items():
            if mag_col not in row.index:
                continue
            app_mag = row[mag_col]
            if not np.isfinite(app_mag):
                continue
            observed[band] = app_mag - mu
            raw_err = row[err_col] if err_col in row.index else np.nan
            errors[band] = max(float(raw_err) if np.isfinite(raw_err) else min_err, min_err)

        bands = [b for b in ACTIVE_BANDS if b in observed and b in _grid and np.isfinite(observed[b])]
        if len(bands) < 3:
            skipped += 1
            continue

        feh_lo   = st_met - FEH_TOLERANCE
        feh_hi   = st_met + FEH_TOLERANCE
        obs_teff = float(st_teff) if np.isfinite(st_teff) else None
        obs_logg = float(st_logg) if np.isfinite(st_logg) else None

        p0_center = np.array([fit_row.mass_msun,
                               np.log10(fit_row.age_gyr * 1e9),
                               fit_row.feh])
        scales = np.array([0.05, 0.05, 0.02])
        p0     = p0_center + scales * np.random.randn(N_WALKERS, 3)
        p0[:, 0] = np.clip(p0[:, 0], 0.08, 3.0)
        p0[:, 1] = np.clip(p0[:, 1], logages[0], logages[-1])
        p0[:, 2] = np.clip(p0[:, 2], feh_lo, feh_hi)

        pool_ctx = Pool(N_CORES) if N_CORES > 1 else None
        sampler  = emcee.EnsembleSampler(
            N_WALKERS, 3, log_prob,
            args=(observed, errors, bands, feh_lo, feh_hi, obs_teff, obs_logg),
            pool=pool_ctx,
        )

        sampler.run_mcmc(p0, N_BURN, progress=False)
        p_burned = sampler.get_last_sample().coords
        sampler.reset()
        sampler.run_mcmc(p_burned, N_STEPS, progress=False)

        if pool_ctx is not None:
            pool_ctx.close()
            pool_ctx.join()

        flat      = sampler.get_chain(flat=True)
        ages_gyr  = 10 ** flat[:, 1] / 1e9
        masses_s  = flat[:, 0]
        fehs_s    = flat[:, 2]
        accept_fr = np.mean(sampler.acceptance_fraction)

        records.append({
            'hostname':   hostname,
            'age_16':     round(np.percentile(ages_gyr, 16), 4),
            'age_50':     round(np.percentile(ages_gyr, 50), 4),
            'age_84':     round(np.percentile(ages_gyr, 84), 4),
            'mass_16':    round(np.percentile(masses_s, 16), 4),
            'mass_50':    round(np.percentile(masses_s, 50), 4),
            'mass_84':    round(np.percentile(masses_s, 84), 4),
            'feh_16':     round(np.percentile(fehs_s, 16), 4),
            'feh_50':     round(np.percentile(fehs_s, 50), 4),
            'feh_84':     round(np.percentile(fehs_s, 84), 4),
            'accept_fr':  round(accept_fr, 3),
            'age_err_lo': round(np.percentile(ages_gyr, 50) - np.percentile(ages_gyr, 16), 4),
            'age_err_hi': round(np.percentile(ages_gyr, 84) - np.percentile(ages_gyr, 50), 4),
        })

        elapsed  = time.time() - _t0
        per_star = elapsed / (i + 1)
        eta_min  = per_star * (n_total - i - 1) / 60
        print(f"  [{i+1}/{n_total}] {hostname:30s}  "
              f"age={np.percentile(ages_gyr,50):.2f} "
              f"[{np.percentile(ages_gyr,16):.2f}–{np.percentile(ages_gyr,84):.2f}] Gyr  "
              f"accept={accept_fr:.2f}  ETA {eta_min:.0f} min", flush=True)

        # Checkpoint every 25 stars
        if (i + 1) % 25 == 0:
            pd.DataFrame(records).to_csv(MCMC_CSV, index=False)
            print(f"  ── checkpoint saved ({len(records)} stars) ──", flush=True)

    mcmc_df = pd.DataFrame(records)
    mcmc_df.to_csv(MCMC_CSV, index=False)
    total_min = (time.time() - _t0) / 60
    print(f"\nDone — {len(mcmc_df)} stars in {total_min:.1f} min  ({skipped} skipped)")
    print(f"Saved to: {MCMC_CSV}")
    print(mcmc_df[['hostname', 'age_16', 'age_50', 'age_84', 'accept_fr']].head(20).to_string(index=False))
