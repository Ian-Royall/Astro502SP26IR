# Claude Grid Gen(T_eff + logg)
import os
import time
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ezpadova import parsec

# ── Grid Parameters ───────────────────────────────────────────────────────────
masses_test  = np.linspace(0.1, 2.8, 100)
fehs         = np.linspace(-1.0, 0.6, 50)

# Non-uniform logAge: coarser over young ages (well-resolved by log scale),
# dense over old ages (>1 Gyr) where linear spacing blows up past 0.25 Gyr/step.
_la_young    = np.linspace(7.5, 9.0, 25)   # 0.03–1 Gyr,  ~0.063 dex spacing
_la_old      = np.linspace(9.0, 10.0, 101) # 1–10 Gyr,   ~0.010 dex spacing
logages_test = np.concatenate([_la_young, _la_old[1:]])  # 124 nodes total

_DIR            = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_FILE = os.path.join(_DIR, 'parsec_grid_checkpoint_v2.npz')
OUTPUT_FILE     = os.path.join(_DIR, 'Tera Grid.npz')

# ── Retry config ──────────────────────────────────────────────────────────────
# Delays (seconds) between retries on connection loss; last value repeats.
_RETRY_DELAYS = [15, 30, 60, 120, 300]

_NETWORK_KEYWORDS = (
    'Max retries exceeded',
    'NameResolutionError',
    'Failed to resolve',
    'ConnectionError',
    'RemoteDisconnected',
    'Connection refused',
    'timeout',
    'Timeout',
)


def _fetch_with_retry(la, feh, photsys_file):
    """
    Call parsec.get_isochrones(), retrying indefinitely on network errors.
    Non-network exceptions propagate immediately so the outer handler can
    log them as data failures rather than connection failures.
    """
    attempt = 0
    while True:
        try:
            return parsec.get_isochrones(
                logage=(la, la, 0.0),
                MH=(feh, feh, 0.0),
                photsys_file=photsys_file,
            )
        except Exception as e:
            if any(kw in str(e) for kw in _NETWORK_KEYWORDS):
                delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                print(f"  [connection lost] retrying in {delay}s "
                      f"(attempt {attempt + 1})…")
                time.sleep(delay)
                attempt += 1
            else:
                raise


def _save_checkpoint(path, mag_grids, teff_grid, logg_grid,
                     completed, masses, logages, fehs):
    completed_arr = (np.array(list(completed), dtype=int)
                     if completed else np.empty((0, 2), dtype=int))
    np.savez(
        path,
        **{f'band_{k}': v for k, v in mag_grids.items()},
        teff_grid=teff_grid,
        logg_grid=logg_grid,
        completed=completed_arr,
        masses=masses,
        logages=logages,
        fehs=fehs,
    )


def _load_checkpoint(path, mag_grids, teff_grid, logg_grid):
    """
    Load checkpoint into the pre-allocated grids in-place.
    Returns the set of already-completed (i_f, i_la) index pairs.
    """
    if not os.path.exists(path):
        return set()
    print(f"Checkpoint found at '{path}' — resuming…")
    ckpt = np.load(path)
    for band in mag_grids:
        key = f'band_{band}'
        if key in ckpt:
            mag_grids[band][:] = ckpt[key]
    teff_grid[:] = ckpt['teff_grid']
    logg_grid[:] = ckpt['logg_grid']
    arr = ckpt['completed']
    completed = set(map(tuple, arr.tolist())) if arr.shape[0] > 0 else set()
    print(f"  {len(completed)} grid points already done, skipping those.")
    return completed


# ── Main generator ────────────────────────────────────────────────────────────
def generate_parsec_grid_test(masses, logages, fehs,
                               checkpoint_file=CHECKPOINT_FILE):
    shape    = (len(fehs), len(logages), len(masses))

    band_map = {
        'G':  ('gaiaEDR3',           'Gmag'),
        'BP': ('gaiaEDR3',           'G_BPmag'),
        'RP': ('gaiaEDR3',           'G_RPmag'),
        'J':  ('2mass_spitzer_wise', 'Jmag'),
        'H':  ('2mass_spitzer_wise', 'Hmag'),
        'K':  ('2mass_spitzer_wise', 'Ksmag'),
        'W1': ('2mass_spitzer_wise', 'W1mag'),
        'W2': ('2mass_spitzer_wise', 'W2mag'),
        'W3': ('2mass_spitzer_wise', 'W3mag'),
        'W4': ('2mass_spitzer_wise', 'W4mag'),
    }

    mag_grids = {band: np.full(shape, np.nan) for band in band_map}
    teff_grid = np.full(shape, np.nan)
    logg_grid = np.full(shape, np.nan)

    # Resume from checkpoint if one exists
    completed = _load_checkpoint(checkpoint_file, mag_grids, teff_grid, logg_grid)

    success_count = len(completed)
    empty_count   = 0
    fail_count    = 0
    total         = len(fehs) * len(logages)

    for i_f, feh in enumerate(fehs):
        for i_la, la in enumerate(logages):

            if (i_f, i_la) in completed:
                continue

            try:
                gaia_table = _fetch_with_retry(la, feh, 'gaiaEDR3')
                nir_table  = _fetch_with_retry(la, feh, '2mass_spitzer_wise')

                if len(gaia_table) == 0 or len(nir_table) == 0:
                    empty_count += 1
                    print(f"Empty table(s) for [M/H]={feh:.3f}, logAge={la:.3f}")
                    completed.add((i_f, i_la))   # don't retry empty isochrones
                    continue

                gaia_table = gaia_table.sort_values('Mini').drop_duplicates('Mini')
                nir_table  = nir_table.sort_values('Mini').drop_duplicates('Mini')

                merged = gaia_table.merge(
                    nir_table,
                    on='Mini',
                    how='inner',
                    suffixes=('', '_nir'),
                )

                if len(merged) == 0:
                    print(f"No overlap after merge for [M/H]={feh:.3f}, logAge={la:.3f}")
                    completed.add((i_f, i_la))
                    continue

                merged = merged.sort_values('Mini')
                mini   = merged['Mini'].values

                unique_idx  = np.unique(mini, return_index=True)[1]
                mini_strict = mini[unique_idx]

                if len(mini_strict) < 2:
                    print(f"Too few unique Mini at feh={feh:.3f}, logAge={la:.3f}")
                    completed.add((i_f, i_la))
                    continue

                sort_idx    = np.argsort(mini_strict)
                mini_strict = mini_strict[sort_idx]

                valid_idx    = np.where(
                    (masses >= mini_strict[0]) & (masses <= mini_strict[-1])
                )[0]
                masses_valid = masses[valid_idx]

                if len(masses_valid) == 0:
                    completed.add((i_f, i_la))
                    continue

                for band, (sys_name, col) in band_map.items():
                    if col not in merged.columns:
                        continue
                    mag_values  = merged[col].values[unique_idx][sort_idx]
                    finite_mask = np.isfinite(mag_values)
                    if finite_mask.sum() < 3:
                        continue
                    mv = mag_values[finite_mask]
                    interp_fn = interp1d(
                        mini_strict[finite_mask],
                        mv,
                        kind='quadratic',
                        bounds_error=False,
                        fill_value=(mv[0], mv[-1]),  # extend boundary, no NaN
                    )
                    mag_grids[band][i_f, i_la, valid_idx] = interp_fn(masses_valid)

                for phys_col, out_grid in [('logTe', teff_grid), ('logg', logg_grid)]:
                    if phys_col not in merged.columns:
                        print(f"Missing column '{phys_col}' — check PARSEC version")
                        continue
                    phys_values = merged[phys_col].values[unique_idx][sort_idx]
                    finite_mask = np.isfinite(phys_values)
                    if finite_mask.sum() < 3:
                        continue
                    pv = phys_values[finite_mask]
                    interp_fn = interp1d(
                        mini_strict[finite_mask],
                        pv,
                        kind='quadratic',
                        bounds_error=False,
                        fill_value=(pv[0], pv[-1]),  # extend boundary, no NaN
                    )
                    interpolated = interp_fn(masses_valid)
                    if phys_col == 'logTe':
                        interpolated = 10 ** interpolated
                    out_grid[i_f, i_la, valid_idx] = interpolated

                success_count += 1
                completed.add((i_f, i_la))
                _save_checkpoint(checkpoint_file, mag_grids, teff_grid, logg_grid,
                                 completed, masses, logages, fehs)

                done = len(completed)
                print(f"[{done}/{total}] feh={feh:.3f}, logAge={la:.3f} ✓")

            except Exception as e:
                fail_count += 1
                print(f"Query/merge failed for [M/H]={feh:.3f}, logAge={la:.3f}: {e}")

    print(f"\nGrid generation complete: {success_count} success, "
          f"{empty_count} empty, {fail_count} failed")

    return mag_grids, masses, logages, fehs, teff_grid, logg_grid


# ── Generate ──────────────────────────────────────────────────────────────────
grid_test, m_test, la_test, fehs_out, teff_grid, logg_grid = generate_parsec_grid_test(
    masses_test, logages_test, fehs
)

# ── Save final output ─────────────────────────────────────────────────────────
np.savez(
    OUTPUT_FILE,
    **grid_test,
    masses=m_test,
    logages=la_test,
    fehs=fehs_out,
    Teff=teff_grid,   # shape (n_feh, n_logage, n_mass), units Kelvin
    logg=logg_grid,   # shape (n_feh, n_logage, n_mass), units dex
)
print(f"Grid saved to '{OUTPUT_FILE}'.")

# Clean up checkpoint now that the final file is written
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("Checkpoint file removed.")

# ── Quick sanity check on Teff/logg at solar params ──────────────────────────
idx_f  = np.argmin(np.abs(fehs_out))
idx_la = np.argmin(np.abs(la_test - np.log10(4.6e9)))
idx_m  = np.argmin(np.abs(m_test  - 1.0))
print(f"\nSolar node check (feh={fehs_out[idx_f]:.3f}, "
      f"logage={la_test[idx_la]:.3f}, mass={m_test[idx_m]:.3f}):")
print(f"  Teff = {teff_grid[idx_f, idx_la, idx_m]:.0f} K  (expect ~5778 K)")
print(f"  logg = {logg_grid[idx_f, idx_la, idx_m]:.3f}    (expect ~4.44)")
