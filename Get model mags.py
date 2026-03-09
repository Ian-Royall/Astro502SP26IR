import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ezpadova import parsec

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
print(np.shape(grid['G']))