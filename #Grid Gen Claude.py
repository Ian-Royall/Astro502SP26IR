import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ezpadova import parsec

# Grid Parameters
masses_test  = np.linspace(0.1, 2.8, 100)
logages_test = np.linspace(7.5, 10.0, 40)
fehs         = np.linspace(-1.0, 0.6, 50)


def generate_parsec_grid_test(masses, logages, fehs):
    """
    Generate 3D magnitude grid by querying Gaia and NIR separately and merging.
    Returns dict of 3D arrays [n_feh, n_logage, n_mass].

    Key fix: grid points with mass > turnoff mass for that (feh, logage) slice
    are left as NaN — which is physically correct — but the interpolation range
    is now clamped to [mini_min, mini_max] returned by PARSEC, so we no longer
    produce spurious NaNs from extrapolation within the valid mass range.
    """
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

    # Also store the valid mass range per slice for diagnostics
    turnoff_mass = np.full((len(fehs), len(logages)), np.nan)  # max Mini returned

    success_count = 0
    empty_count   = 0
    fail_count    = 0

    for i_f, feh in enumerate(fehs):
        for i_la, la in enumerate(logages):
            try:
                # ── Query both photometric systems ──────────────────────────
                gaia_table = parsec.get_isochrones(
                    logage=(la, la, 0.0),
                    MH=(feh, feh, 0.0),
                    photsys_file='gaiaEDR3'
                )
                nir_table = parsec.get_isochrones(
                    logage=(la, la, 0.0),
                    MH=(feh, feh, 0.0),
                    photsys_file='2mass_spitzer_wise'
                )

                if len(gaia_table) == 0 or len(nir_table) == 0:
                    empty_count += 1
                    print(f"Empty table(s) for [M/H]={feh:.3f}, logAge={la:.3f}")
                    continue

                # ── Sort + dedup on Mini ────────────────────────────────────
                gaia_table = gaia_table.sort_values('Mini').drop_duplicates('Mini')
                nir_table  = nir_table.sort_values('Mini').drop_duplicates('Mini')

                merged = gaia_table.merge(
                    nir_table,
                    on=['Mini', 'logAge', 'MH'],
                    how='inner'
                )

                if len(merged) == 0:
                    print(f"No overlap after merge for [M/H]={feh:.3f}, logAge={la:.3f}")
                    continue

                merged = merged.sort_values('Mini')
                mini   = merged['Mini'].values

                unique_idx   = np.unique(mini, return_index=True)[1]
                mini_strict  = mini[unique_idx]

                if len(mini_strict) < 2:
                    print(f"Too few unique Mini at feh={feh:.3f}, logAge={la:.3f}")
                    continue

                # Guarantee strictly increasing
                sort_idx    = np.argsort(mini_strict)
                mini_strict = mini_strict[sort_idx]

                # ── FIX: record the valid mass range for this slice ─────────
                mini_min = mini_strict[0]
                mini_max = mini_strict[-1]
                turnoff_mass[i_f, i_la] = mini_max

                # ── FIX: only interpolate masses within [mini_min, mini_max]
                #    Use integer indices (not boolean mask) to avoid numpy
                #    fancy-indexing returning a 1D view that confuses scipy.
                #    Masses outside this range stay NaN (physically correct).
                valid_idx    = np.where((masses >= mini_min) & (masses <= mini_max))[0]
                masses_valid = masses[valid_idx]

                if len(masses_valid) == 0:
                    continue

                success_count += 1

                for band, (sys_name, col) in band_map.items():
                    if col not in merged.columns:
                        print(f"Missing column '{col}' at feh={feh:.3f}, logAge={la:.3f}")
                        continue

                    mag_values = merged[col].values[unique_idx][sort_idx]

                    # Drop any NaN magnitudes in the isochrone itself before interpolating
                    finite_mask = np.isfinite(mag_values)
                    if finite_mask.sum() < 2:
                        continue

                    interp_fn = interp1d(
                        mini_strict[finite_mask],
                        mag_values[finite_mask],
                        kind='linear',
                        bounds_error=False,
                        fill_value=np.nan,
                    )

                    mag_grids[band][i_f, i_la, valid_idx] = interp_fn(masses_valid)

            except Exception as e:
                fail_count += 1
                print(f"Query/merge failed for [M/H]={feh:.3f}, logAge={la:.3f}: {e}")

    print(f"\nGrid generation complete: {success_count} success, "
          f"{empty_count} empty, {fail_count} failed")

    return mag_grids, masses, logages, fehs, turnoff_mass


# ── Generate and save ───────────────────────────────────────────────────────
grid_test, m_test, la_test, fehs_out, turnoff = generate_parsec_grid_test(
    masses_test, logages_test, fehs
)

np.savez(
    'parsec_grid_clipped_claude(2.8).npz',
    **grid_test,
    masses=m_test,
    logages=la_test,
    fehs=fehs_out,
    turnoff_mass=turnoff,   # saved for diagnostics / fitting masks
)

print("Grid saved.")


# ── Quick NaN diagnostic after generation ──────────────────────────────────
for band in ['G', 'J']:
    print(f"\nNaN vs logage — band: {band}")
    for i_la, la in enumerate(la_test):
        n_nan = int(np.sum(np.isnan(grid_test[band][:, i_la, :])))
        total = grid_test[band][:, i_la, :].size
        pct   = 100 * n_nan / total
        print(f"  logage={la:.2f}: {n_nan}/{total} NaNs ({pct:.1f}%)")
    print(f"\nNaN vs logage — band: {band}")
    for i_la, la in enumerate(la_test):
        n_nan = int(np.sum(np.isnan(grid_test[band][:, i_la, :])))
        total = grid_test[band][:, i_la, :].size
        pct   = 100 * n_nan / total
        print(f"  logage={la:.2f}: {n_nan}/{total} NaNs ({pct:.1f}%)")