import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ezpadova import parsec

# Grid Parameters
masses_test = np.linspace(0.1, 2.8, 100)  # 100 mass points from 0.1 to 2.8 Msun
logages_test = np.linspace(7.5, 10.0, 40)  # 40 points, focused on useful ages
fehs = np.linspace(-1.0, 0.6, 50)   # 50 metallicities to see patterns

def generate_parsec_grid_test(masses, logages, fehs):
    """
    Generate 3D magnitude grid by querying Gaia and NIR separately and merging.
    Returns dict of 3D arrays [n_feh, n_logage, n_mass]
    """
    shape = (len(fehs), len(logages), len(masses))
    mag_grids = {}

    # All desired bands + their column names in each system
    band_map = {
        'G':   ('gaiaEDR3', 'Gmag'),
        'BP':  ('gaiaEDR3', 'G_BPmag'),
        'RP':  ('gaiaEDR3', 'G_RPmag'),
        'J':   ('2mass_spitzer_wise', 'Jmag'),
        'H':   ('2mass_spitzer_wise', 'Hmag'),
        'K':   ('2mass_spitzer_wise', 'Ksmag'),  # Ks in 2MASS
        'W1':  ('2mass_spitzer_wise', 'W1mag'),
        'W2':  ('2mass_spitzer_wise', 'W2mag'),
        'W3':  ('2mass_spitzer_wise', 'W3mag'),
        'W4':  ('2mass_spitzer_wise', 'W4mag')
    }

    # Initialize empty grids
    for band in band_map:
        mag_grids[band] = np.full(shape, np.nan)

    success_count = 0
    empty_count = 0
    fail_count = 0

    for i_f, feh in enumerate(fehs):
        for i_la, la in enumerate(logages):
            try:
                # Query Gaia
                gaia_table = parsec.get_isochrones(
                    logage=(la, la, 0.0),
                    MH=(feh, feh, 0.0),
                    photsys_file='gaiaEDR3'
                )

                # Query NIR
                nir_table = parsec.get_isochrones(
                    logage=(la, la, 0.0),
                    MH=(feh, feh, 0.0),
                    photsys_file='2mass_spitzer_wise'
                )

                if len(gaia_table) == 0 or len(nir_table) == 0:
                    empty_count += 1
                    print(f"Empty table(s) for [M/H]={feh:.3f}, logAge={la:.3f}")
                    continue

                # Sort and dedup both tables
                gaia_table = gaia_table.sort_values('Mini').drop_duplicates('Mini')
                nir_table = nir_table.sort_values('Mini').drop_duplicates('Mini')

                # Merge on common keys
                merged = gaia_table.merge(
                    nir_table,
                    on=['Mini', 'logAge', 'MH'],
                    how='inner'
                )

                if len(merged) == 0:
                    print(f"No overlap after merge for [M/H]={feh:.3f}, logAge={la:.3f}")
                    continue

                success_count += 1

                # Use merged table
                merged = merged.sort_values('Mini')
                mini = merged['Mini'].values

                # Strict Mini handling
                unique_idx = np.unique(mini, return_index=True)[1]
                mini_strict = mini[unique_idx]

                if len(mini_strict) < 2:
                    print(f"Too few unique Mini after dedup at feh={feh:.3f}, logAge={la:.3f}")
                    continue

                if np.any(np.diff(mini_strict) <= 0):
                    sort_idx = np.argsort(mini_strict)
                    mini_strict = mini_strict[sort_idx]
                else:
                    sort_idx = np.arange(len(mini_strict))

                # Fill from merged table
                for band, (sys_name, col) in band_map.items():
                    if col in merged.columns:
                        mag_values = merged[col].values[unique_idx]
                        mag_values = mag_values[sort_idx] if 'sort_idx' in locals() else mag_values
                        interp = interp1d(
                            mini_strict, mag_values,
                            kind='linear',
                            bounds_error=False,
                            fill_value=np.nan
                        )
                        mag_grids[band][i_f, i_la] = interp(masses)
                    else:
                        print(f"Missing column '{col}' after merge at feh={feh:.3f}, logAge={la:.3f}")

            except Exception as e:
                fail_count += 1
                print(f"Query/merge failed for [M/H]={feh:.3f}, logAge={la:.3f}: {e}")
    return mag_grids, masses, logages, fehs
    
# Test Grid Generation
grid_test, m_test, la_test, fehs = generate_parsec_grid_test(masses_test, logages_test, fehs)
#save test grid
np.savez('parsec_grid_clipped(2.8).npz', **grid_test, masses=m_test, logages=la_test, fehs=fehs)
