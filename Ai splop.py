import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ezpadova import parsec

def generate_parsec_grid(masses, logages, fehs, phot_system='gaiaEDR3'):
    """
    Generate 3D magnitude grid directly (no 1D interpolation in loop).
    Returns dict of 3D arrays [n_feh, n_logage, n_mass]
    """
    shape = (len(fehs), len(logages), len(masses))
    mag_grids = {}

    # Bands we care about + expected column names
    band_map = {
        'G': 'Gmag',
        'BP': 'G_BPmag',
        'RP': 'G_RPmag',
        'J': 'Jmag',
        'H': 'Hmag',
        'K': 'Kmag'   # or 'Ksmag' if 2MASS
        # Add 'g', 'r', 'i', 'z', 'W1', 'W2', 'W3', 'W4' if your system has them
    }

    # Initialize empty grids
    for band in band_map:
        mag_grids[band] = np.full(shape, np.nan)

    for i_f, feh in enumerate(fehs):
        for i_la, la in enumerate(logages):
            try:
                table = parsec.get_isochrones(
                    logage=(la, la, 0.0),
                    MH=(feh, feh, 0.0),
                    photsys_file=phot_system
                )
                if len(table) == 0:
                    print(f"Empty table for [M/H]={feh}, logAge={la}")
                    continue
                # Pandas sort + drop duplicates for strict ascending Mini
                table = table.sort_values('Mini').drop_duplicates('Mini')
                mini = table['Mini'].values

                table.sort_values('Mini')
                mini = table['Mini'].values  # numpy array

                # Fill each band if column exists
                for band, col in band_map.items():
                    if col in table.columns:
                        # Interpolate to our desired mass grid
                        interp = RegularGridInterpolator(
                            (mini,),
                            table[col].values,
                            bounds_error=False,
                            fill_value=np.nan
                        )
                        mag_grids[band][i_f, i_la] = interp(masses)
                    else:
                        print(f"Missing column '{col}' in system {phot_system}")
            except Exception as e:
                print(f"Query failed for [M/H]={feh}, logAge={la}: {e}")

    return mag_grids, masses, logages, fehs


# Example usage (small grid for testing)
masses = np.linspace(0.5, 3.0, 20)
logages = np.linspace(9.0, 9.7, 5)   # ~1–5 Gyr
fehs = np.array([-0.5, 0.0, 0.3])

grid, masses, logages, fehs = generate_parsec_grid(masses, logages, fehs)

# Save for later 
np.savez('parsec_grid.npz', **grid, masses=masses, logages=logages, fehs=fehs)


# ──────────────────────────────────────────
# Final get_model_mag using 3D interpolation
# ──────────────────────────────────────────

def get_model_mag(mass, age_yr, feh, grid, masses, logages, fehs):
    """
    Interpolate from pre-generated 3D grid.
    age_yr: age in years (e.g. 2e9 for 2 Gyr)
    """
    logage = np.log10(age_yr)

    mags = {}
    for band in grid:
        interpolator = RegularGridInterpolator(
            (fehs, logages, masses),
            grid[band],
            bounds_error=False,
            fill_value=np.nan
        )
        mags[band] = interpolator((feh, logage, mass))
    return mags


# Load saved grid (after generation)
data = np.load('parsec_grid.npz')
grid = {k: data[k] for k in data if k not in ['masses', 'logages', 'fehs']}
masses = data['masses']
logages = data['logages']
fehs = data['fehs']

# Test
raw_mags = get_model_mag(mass=1.0, age_yr=4.57e9, feh=0.0, grid=grid, masses=masses, logages=logages, fehs=fehs)
print("Raw Solar Model magnitudes:", raw_mags)

# Calibration

calibrated_mags = {}
calibrated_mags['G']   = raw_mags['G']   + 0.12  # Adjust G mag for solar model to match observed solar G 
calibrated_mags['BP']  = raw_mags['BP']
calibrated_mags['RP']  = raw_mags['RP']  + 0.19  # Adjust RP mag for solar model to match observed solar RP

print("Calibrated Solar Model magnitudes:", calibrated_mags)