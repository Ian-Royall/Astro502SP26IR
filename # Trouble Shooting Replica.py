# Trouble Shooting Replica
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RegularGridInterpolator
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

# ──────────────────────────────────────────
# Brute-force best-fit + refine
# ──────────────────────────────────────────
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
'''
def refine_fit(observed, errors, grid, masses, logages, fehs, rough_params, zoom_factor=0.5):
    rough_mass, rough_age_yr, rough_feh = rough_params
    # Zoom ranges
    m_zoom = np.linspace(rough_mass * (1 - zoom_factor), rough_mass * (1 + zoom_factor), 80)
    la_zoom = np.linspace(np.log10(rough_age_yr) - 0.3, np.log10(rough_age_yr) + 0.3, 100)
    f_zoom = np.linspace(rough_feh - 0.3, rough_feh + 0.3, 20)
    # Re-run fit with zoomed grid
    return brute_force_best_fit(observed, errors, grid, m_zoom, la_zoom, f_zoom)
'''

# Load saved grid (small)
data1 = np.load('parsec_grid_test.npz')  # change to 'parsec_grid_full_bands.npz' after generation
grid1 = {k: data1[k] for k in data1 if k not in ['masses', 'logages', 'fehs']}
masses1 = data1['masses']
logages1 = data1['logages']
fehs1 = data1['fehs']

# Laod saved grid (LARGE)
data2 = np.load('parsec_grid_with_NIR.npz')
grid2 = {k: data2[k] for k in data2 if k not in ['masses', 'logages', 'fehs']}
masses2 = data2['masses']
logages2 = data2['logages']
fehs2 = data2['fehs']
# ──────────────────────────────────────────
# Example Sun test with all bands
# ──────────────────────────────────────────
observed_sun = {
    'G':   4.65,
    'BP':  4.83,
    'RP':  4.24,
    'J':   3.64,
    'H':   3.32,
    'K':   3.28,
    'W1':  3.2,
    'W2':  3.3,
    'W3':  3.4,
    'W4':  3.5
}

errors_sun = {
    'G':   0.01,
    'BP':  0.02,
    'RP':  0.02,
    'J':   0.03,
    'H':   0.03,
    'K':   0.03,
    'W1':  0.03,
    'W2':  0.03,
    'W3':  0.03,
    'W4':  0.03
}


bands_full = [b for b in observed_sun if b in grid1]
print("Fitting with bands:", bands_full)

# For checking reasons
band = 'G'  # or 'J' to check NIR

# Grid 1 test
rough_params, rough_chi2 = brute_force_best_fit(observed_sun, errors_sun, grid1, masses1, logages1, fehs1, bands=bands_full)
print("Rough fit Grid 1:", rough_params, rough_chi2)

nan_per_slice = np.sum(np.isnan(grid1[band]), axis=2)  # sum over masses
print("Grid 1 Nan Test, NaNs per (feh, logage) slice (max possible = len(masses1))")
print("Average NaNs per slice:", nan_per_slice.mean())
print("Slices with full NaNs:", np.sum(nan_per_slice == len(masses1)))
print("Slices with partial NaNs:", np.sum((nan_per_slice > 0) & (nan_per_slice < len(masses1))))
print("Slices with no NaNs:", np.sum(nan_per_slice == 0))
print("shape:", np.shape(data1['G']))  # should be (n_feh, n_logage, n_mass)

# Grid 2 
rough_params, rough_chi2 = brute_force_best_fit(observed_sun, errors_sun, grid2, masses2, logages2, fehs2, bands=bands_full)
print("Rough fit Grid 2:", rough_params, rough_chi2)

nan_per_slice = np.sum(np.isnan(grid2[band]), axis=2)  # sum over masses
print("Grid 2 Nan Test, NaNs per (feh, logage) slice (max possible = len(masses2))")
print("Average NaNs per slice:", nan_per_slice.mean())
print("Slices with full NaNs:", np.sum(nan_per_slice == len(masses2)))
print("Slices with partial NaNs:", np.sum((nan_per_slice > 0) & (nan_per_slice < len(masses2))))
print("Slices with no NaNs:", np.sum(nan_per_slice == 0))
print("shape:", np.shape(data2['G']))  # should be (n_feh, n_logage, n_mass)


#Clipping test Grid 1
print("\nClipping test for Grid 1:")

safe_max = 2.9  # conservative — adjust to 7.0 if needed
mask = masses1 <= safe_max
masses_clip1 = masses1[mask]
grid_clip1 = {band: grid1[band][..., mask] for band in grid1}

print(f"Clipped to {safe_max} M⊙: {len(masses_clip1)} masses remain")
print("New shape per band:", list(grid_clip1.values())[0].shape)

# test output post clip 1
# Use clipped masses array so indices match the clipped grid
clip_params1, clip_chi21 = brute_force_best_fit(observed_sun, errors_sun, grid_clip1, masses_clip1, logages1, fehs1, bands=bands_full)
print("Clipped fit Grid 1:", clip_params1, clip_chi21)

# NaN check after clip
for band in grid_clip1:
    nan_count = np.sum(np.isnan(grid_clip1[band]))
    total = grid_clip1[band].size
    percent = 100 * nan_count / total
    print(f"{band}: {percent:.1f}% NaN after clip")






print("\nClipping test for Grid 2:")
# Clip to safe upper limit (based on your diagnostic: average max = 6.24 M⊙)
safe_max = 2.9  # conservative — adjust to 7.0 if needed
mask = masses2 <= safe_max
masses_clip2 = masses2[mask]
grid_clip2 = {band: grid2[band][..., mask] for band in grid2}

print(f"Clipped to {safe_max} M⊙: {len(masses_clip2)} masses remain")
print("New shape per band:", list(grid_clip2.values())[0].shape)

# test output post clip 2
# Use clipped masses array so indices match the clipped grid
clip_params2, clip_chi22 = brute_force_best_fit(observed_sun, errors_sun, grid_clip2, masses_clip2, logages2, fehs2, bands=bands_full)
print("Clipped fit Grid 2:", clip_params2, clip_chi22)

# NaN check after clip
for band in grid_clip2:
    nan_count = np.sum(np.isnan(grid_clip2[band]))
    total = grid_clip2[band].size
    percent = 100 * nan_count / total
    print(f"{band}: {percent:.1f}% NaN after clip")
