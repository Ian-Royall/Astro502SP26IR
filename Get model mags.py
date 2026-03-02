import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ezpadova import parsec

def generate_parsec_grid(masses, logages, fehs, phot_system='gaiaEDR3'):

    # Generate 3D magnitude grid directly (no 1D interpolation in loop).
    # Returns dict of 3D arrays [n_feh, n_logage, n_mass]

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
masses = np.linspace(0.08, 12.0, 150)  # from 0.08 to 12 M⊙
logages = np.linspace(6.0, 10.13, 100)   # logAge from 1 Myr to ~13.5 Gyr
fehs = np.linspace(-2.0, 0.6, 25)  # [M/H] = -1.5, 0.0, +0.5
#grid, masses, logages, fehs = generate_parsec_grid(masses, logages, fehs)

# Save for later 
#np.savez('parsec_grid.npz', **grid, masses=masses, logages=logages, fehs=fehs)


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
data = np.load('parsec_grid_with_NIR.npz')
grid = {k: data[k] for k in data if k not in ['masses', 'logages', 'fehs']}
masses = data['masses']
logages = data['logages']
fehs = data['fehs']

# Test
raw_mags = get_model_mag(mass=1.0, age_yr=4.603e9, feh=0.0, grid=grid, masses=masses, logages=logages, fehs=fehs)
print("Raw Solar Model magnitudes:", raw_mags)


# Full loop over all stars (example)
# Assume your data is a list of dicts or pandas DataFrame
# Example: list of dicts
stars = [
    {'G': 4.5279774, 'BP': 4.84438424, 'RP': 4.04472947, 'G_err': 0.02, 'BP_err': 0.03, 'RP_err': 0.02},
    # ... add all your stars here
    # or load from CSV: pd.read_csv('your_data.csv').to_dict('records')
]
'''
best_fits = []
for star in stars:
    observed = {b: star[b] for b in ['G', 'BP', 'RP']}
    errors   = {b: star[b+'_err'] for b in ['G', 'BP', 'RP']}
    #test_params, test_chi2 = brute_force_best_fit(observed, errors, grid, masses, logages, fehs)
    params, chi2 = brute_force_best_fit(observed, errors, grid, masses, logages, fehs)
    best_fits.append({
        'mass': params[0],
        'age_yr': params[1],
        'age_Gyr': params[1]/1e9,
        'feh': params[2],
        'chi2': chi2
    })
    print(f"Star: {observed}, Best fit: {params}, χ² = {chi2:.2f}")

Young star example (approximate solar-mass at 100 Myr)'''



def brute_force_best_fit(observed, errors, grid, masses, logages, fehs, bands=None):
    if bands is None:
        bands = list(observed.keys())
    
    best_chi2 = np.inf
    best_idx = (0, 0, 0)
    
    for i_f in range(len(fehs)):
        for i_la in range(len(logages)):
            for i_m in range(len(masses)):
                chi2 = 0.0
                n_valid = 0
                
                for band in bands:
                    model_val = grid[band][i_f, i_la, i_m]
                    if not np.isnan(model_val):
                        chi2 += ((observed[band] - model_val) / errors[band]) ** 2
                        n_valid += 1
                
                if n_valid > 0 and chi2 < best_chi2:
                    best_chi2 = chi2
                    best_idx = (i_m, i_la, i_f)
    
    best_mass = masses[best_idx[0]]
    best_age_yr = 10**logages[best_idx[1]]
    best_feh = fehs[best_idx[2]]
    
    return (best_mass, best_age_yr, best_feh), best_chi2


def refine_fit(observed, errors, grid, masses, logages, fehs, rough_params, zoom_factor=0.5):
    rough_mass, rough_age_yr, rough_feh = rough_params
    
    # Zoom ranges
    m_zoom = np.linspace(rough_mass * (1 - zoom_factor), rough_mass * (1 + zoom_factor), 80)
    la_zoom = np.linspace(np.log10(rough_age_yr) - 0.3, np.log10(rough_age_yr) + 0.3, 100)
    f_zoom = np.linspace(rough_feh - 0.3, rough_feh + 0.3, 20)
    
    # Re-run fit with zoomed grid
    return brute_force_best_fit(observed, errors, grid, m_zoom, la_zoom, f_zoom)


# Example: Sun test
observed_sun = {'G': 4.65, 'BP': 4.83, 'RP': 4.24, 'J': 3.64, 'H': 3.32, 'K': 3.28, 'W1': 3.2, 'W2': 3.3, 'W3': 3.4, 'W4': 3.5}
errors_sun   = {'G': 0.01, 'BP': 0.02, 'RP': 0.02, 'J': 0.03, 'H': 0.03, 'K': 0.03, 'W1': 0.03, 'W2': 0.03, 'W3': 0.03, 'W4': 0.03}

# Stage 1: coarse
rough_params, rough_chi2 = brute_force_best_fit(observed_sun, errors_sun, grid, masses, logages, fehs)
print("Rough fit:", rough_params, rough_chi2)

# Stage 2: refine
refined_params, refined_chi2 = refine_fit(observed_sun, errors_sun, grid, masses, logages, fehs, rough_params)

print("Refined fit:")
print(f"  Mass: {refined_params[0]:.3f} M⊙")
print(f"  Age: {refined_params[1]/1e9:.2f} Gyr")
print(f"  [Fe/H]: {refined_params[2]:.3f}")
print(f"  χ²: {refined_chi2:.2f}")