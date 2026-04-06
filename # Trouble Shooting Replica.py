# Trouble Shooting Replica
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, RegularGridInterpolator
from ezpadova import parsec 
import numpy as np
from scipy.optimize import minimize

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
            fill_value=np.nan,
            method='linear',  # or 'linear' if cubic fails

        )
        mags[band] = interpolator((feh, logage, mass))
    return mags


# Load saved grid 
data2 = np.load('parsec_grid_clipped(2.8nc).npz')  # change to 'parsec_grid_full_bands.npz' after generation
grid2 = {k: data2[k] for k in data2 if k not in ['masses', 'logages', 'fehs']}
masses2 = data2['masses']
logages2 = data2['logages']
fehs2 = data2['fehs']

# Function DEfinitions
def brute_force_best_fit(observed, errors, grid, masses, logages, fehs, bands=None):
    """
    Fast initial grid search to find a good starting point.
    Returns best grid point and χ².
    """
    if bands is None:
        bands = list(observed.keys())
    
    best_chi2 = np.inf
    best_params = (np.nan, np.nan, np.nan)
    
    n_f  = len(fehs)
    n_la = len(logages)
    n_m  = len(masses)
    
    for i_f in range(n_f):
        for i_la in range(n_la):
            for i_m in range(n_m):
                chi2 = 0.0
                n_valid = 0
                
                for band in bands:
                    if band in grid:
                        model_val = grid[band][i_f, i_la, i_m]
                        if not np.isnan(model_val):
                            chi2 += ((observed[band] - model_val) / errors[band]) ** 2
                            n_valid += 1
                
                if n_valid >= 4 and chi2 < best_chi2:  # require at least 4 good bands
                    best_chi2 = chi2
                    best_params = (masses[i_m], 10**logages[i_la], fehs[i_f])
    
    return best_params, best_chi2

def continuous_fit(observed, errors, rough_params, grid, masses, logages, fehs, bands=None, method='Nelder-Mead'):
    """
    Continuous optimization over full parameter space.
    Starts from solar-like guess.
    """
    if bands is None:
        bands = list(observed.keys())

    def chi2_func(params):
        mass, logage, feh = params
        mags = get_model_mag(mass, 10**logage, feh, grid, masses, logages, fehs)
        chi2 = 0.0
        n_valid = 0
        for band in bands:
            if band in mags and not np.isnan(mags[band]):
                diff = observed[band] - mags[band]
                chi2 += diff**2 / errors[band]**2
                n_valid += 1
        if n_valid < 5:  # require at least 5 good bands
            return np.inf
        return chi2 / n_valid  # normalize by valid bands

    # Reasonable starting guess: solar-like
    initial_guess = np.array(rough_params)  # mass, logage, [Fe/H]

    # Bounds: wide but safe
    bounds = [
        (0.08, 3.0),          # mass
        (7.5, 10.2),          # logage (~10 Myr to 15 Gyr)
        (-2.0, 0.6)           # [Fe/H]
    ]

    result = minimize(
        chi2_func,
        initial_guess,
        bounds=bounds,
        method=method,
        options={'maxiter': 1000, 'fatol': 1e-6, 'xatol': 1e-6, 'disp': False}
    )

    if result.success:
        best_mass, best_logage, best_feh = result.x
        best_age_yr = 10**best_logage
        print("Continuous fit success!")
        print(f"  Mass: {best_mass:.3f} M⊙")
        print(f"  Age: {best_age_yr / 1e9:.3f} Gyr")
        print(f"  [Fe/H]: {best_feh:.3f}")
        print(f"  χ²: {result.fun:.3f}")
        print(f"  Message: {result.message}")
    else:
        print("Optimization failed:", result.message)
        print("χ² at initial guess:", chi2_func(initial_guess))
        return initial_guess, chi2_func(initial_guess)

    return (best_mass, best_age_yr, best_feh), result.fun

def continuous_fit_with_perturbations(observed, errors, rough_params, grid, masses, logages, fehs, bands=None, n_starts=8, method='Nelder-Mead'):
    if bands is None:
        bands = list(observed.keys())

    def chi2_func(params):
        mass, logage, feh = params
        mags = get_model_mag(mass, 10**logage, feh, grid, masses, logages, fehs)
        chi2 = 0.0
        n_valid = 0
        for band in bands:
            if band in mags and not np.isnan(mags[band]):
                diff = observed[band] - mags[band]
                chi2 += diff**2 / errors[band]**2
                n_valid += 1
        if n_valid < 5:
            return np.inf
        return chi2 / n_valid

    rough_mass, rough_age_yr, rough_feh = rough_params

    # Base bounds around rough point
    base_bounds = [
        (max(0.08, rough_mass * 0.7), min(8.0, rough_mass * 1.3)),
        (max(7.0, np.log10(rough_age_yr) - 0.8), min(10.2, np.log10(rough_age_yr) + 0.8)),
        (max(-2.0, rough_feh - 0.8), min(0.6, rough_feh + 0.8))
    ]

    best_chi2 = np.inf
    best_result = None

    for start in range(n_starts):
        # Perturb initial guess
        perturbation = np.random.uniform(-0.15, 0.15, 3)  # wider for logage/feh
        perturbed = np.array([rough_mass, np.log10(rough_age_yr), rough_feh]) * (1 + perturbation)
        
        # Clip to bounds
        perturbed = np.clip(perturbed, [b[0] for b in base_bounds], [b[1] for b in base_bounds])

        print(f"Start {start+1}/{n_starts}: initial = {perturbed}")

        result = minimize(
            chi2_func,
            perturbed,
            bounds=base_bounds,
            method=method,
            options={'maxiter': 1000, 'fatol': 1e-6, 'xatol': 1e-6, 'disp': False}
        )

        if result.fun < best_chi2:
            best_chi2 = result.fun
            best_result = result

    if best_result is not None and best_result.success:
        best_mass, best_logage, best_feh = best_result.x
        best_age_yr = 10**best_logage
        print("Best continuous fit across starts:")
        print(f"  Mass: {best_mass:.3f} M⊙")
        print(f"  Age: {best_age_yr / 1e9:.3f} Gyr")
        print(f"  [Fe/H]: {best_feh:.3f}")
        print(f"  χ²: {best_chi2:.3f}")
        print(f"  Message: {best_result.message}")
        return (best_mass, best_age_yr, best_feh), best_chi2
    else:
        print("All starts failed or did not converge")
        return tuple(perturbed), chi2_func(perturbed)

def inspect_best_fit_photometry(observed, params, grid, masses, logages, fehs, bands, label=""):
    mass, age_yr, feh = params
    model_mags = get_model_mag(mass, age_yr, feh, grid, masses, logages, fehs)
    
    print(f"\n=== {label} ===")
    print(f"Forced parameters: mass={mass:.3f} M⊙, age={age_yr/1e6:.0f} Myr, [Fe/H]={feh:.3f}")
    print("Band   Obs     Model    Residual    χ² term")
    print("---------------------------------------------")
    total_chi2 = 0.0
    for band in bands:
        if band in observed and band in model_mags:
            obs = observed[band]
            mod = model_mags[band]
            if np.isnan(mod):
                print(f"{band:4}   {obs:.3f}     NaN       —")
                continue
            res = obs - mod
            err = errors_sun.get(band, 0.03) if "sun" in label.lower() else errors_young1.get(band, 0.03)
            chi_term = (res / err)**2
            total_chi2 += chi_term
            print(f"{band:4}   {obs:.3f}   {mod:.3f}   {res:+7.3f}    {chi_term:.2f}")
    print(f"Total χ²: {total_chi2:.2f}\n")

#Measured Values

#Solar values from Casagrande+2010, with typical errors
observed_sun = {
    'G': 4.65,
   # 'BP': 4.83,
   # 'RP': 4.24,
   # 'J': 3.64,
   # 'H': 3.32,
    'K': 3.28,
    'W1': 3.25,
    'W2': 3.24,
    'W3': 3.23,
   # 'W4': 3.22
} 
errors_sun = {
    'G': 0.01,
   # 'BP': 0.02,
   # 'RP': 0.02,
    #'J': 0.03,
    #'H': 0.03,
    'K': 0.03,
    'W1': 0.03,
    'W2': 0.03,
    'W3': 0.03,
    #'W4': 0.03
}

# Young star test values for Pleiades HD 283518 (V987 Tau) from Gaia DR3, with estimated errors
observed_young1 = {
    'G': 5.85,
    'BP': 6.15,
    'RP': 5.35,
    #'J': 4.95,
    #'H': 4.65,
    'K': 4.55,
    'W1': 4.50,
    'W2': 4.45,
    'W3': 4.40,
    #'W4': 4.35
}
errors_young1 = {
    'G': 0.02,
    'BP': 0.03,
    'RP': 0.02,
    #'J': 0.03,
    #'H': 0.03,
    'K': 0.03,
    'W1': 0.03,
    'W2': 0.03,
    'W3': 0.03,
    #'W4': 0.03
}

# Young star test values for Hyades HD 283518 (V471 Tau) from Gaia DR3, with estimated errors
observed_young2 = {
    'G':  5.15,
    'BP': 5.45,
    'RP': 4.75,
   # 'J':  4.35,
   # 'H':  4.05,
    'K':  3.95,
    'W1': 3.90,
    'W2': 3.85,
    'W3': 3.80,
   # 'W4': 3.75
}
errors_young2 = {
    'G':  0.02,
    'BP': 0.03,
    'RP': 0.02,
  #  'J':  0.03,
  #  'H':  0.03,
    'K':  0.03,
    'W1': 0.03,
    'W2': 0.03,
    'W3': 0.03,
  #  'W4': 0.03
}



# Run the Sun test with continuous and rough fits

#solar test fits with bands G Bp Rp J H K, w1-4 are available but could cause worse fit.
bands_full = [b for b in observed_sun if b in grid2]
print("Fitting with bands:", bands_full)

bands = [b for b in observed_sun if b in grid2]

# Step 1: quick grid search for starting point
rough_params, rough_chi2 = brute_force_best_fit(
    observed_sun, errors_sun, grid2, masses2, logages2, fehs2, bands=bands
)

print("Rough fit Grid 2:", rough_params, rough_chi2)

#contiunous fit starting from rough grid point
best_params, best_chi2 = continuous_fit(
    observed_sun,
    errors_sun,
    rough_params,
    grid2,
    masses2,
    logages2,
    fehs2,
    bands=bands_full,
    method='Nelder-Mead'  # robust to noise/NaNs
)

# Run perturbation tests to check for local minima
best_params_per, best_chi2_per = continuous_fit_with_perturbations(
    observed_sun,
    errors_sun,
    rough_params,  # from your rough fit
    grid2,
    masses2,
    logages2,
    fehs2,
    bands=bands_full,
    n_starts=8  # try 8–10 starts
)

print("Best fit with perturbations:", best_params_per, best_chi2_per)


#--------------analytic checks-------------------

band = 'J'  # or 'J' to check NIR

# Check NaN distribution in grid2 for the chosen band
nan_per_slice = np.sum(np.isnan(grid2[band]), axis=2)  # sum over masses
print("Grid 2 Nan Test, NaNs per (feh, logage) slice (max possible = len(masses2))")
print("Average NaNs per slice:", nan_per_slice.mean())
print("Slices with full NaNs:", np.sum(nan_per_slice == len(masses2)))
print("Slices with partial NaNs:", np.sum((nan_per_slice > 0) & (nan_per_slice < len(masses2))))
print("Slices with no NaNs:", np.sum(nan_per_slice == 0))
print("shape:", np.shape(data2['G']))  # should be (n_feh, n_logage, n_mass)

# ---------------young star tests-------------------

#     Star 1
# Rough fit for young star Pleiades HD 283518
rough_params_young1, rough_chi2_young1 = brute_force_best_fit(
    observed_young1, errors_young1, grid2, masses2, logages2, fehs2, bands=bands
)
print("Rough fit Young Star:", rough_params_young1, rough_chi2_young1)
# Continuous fit for young star Pleiades HD 283518
best_params_young1, best_chi2_young1 = continuous_fit(
    observed_young1,
    errors_young1,
    rough_params_young1,                                                         
    grid2,
    masses2,
    logages2,
    fehs2,
    bands=bands_full,
    method='Nelder-Mead'
)


#     Star 2
# Rough fit for young star Hyades HD 283518
rough_params_young2, rough_chi2_young2 = brute_force_best_fit(
    observed_young2, errors_young2, grid2, masses2, logages2, fehs2, bands=bands
)
print("Rough fit Young Star 2:", rough_params_young2, rough_chi2_young2)
# Continuous fit for young star Hyades HD 283518
best_params_young2, best_chi2_young2 = continuous_fit(
    observed_young2,
    errors_young2,
    rough_params,        
    grid2,
    masses2,
    logages2,
    fehs2,
    bands=bands_full,
    method='Nelder-Mead'
)



#Photometry inspection for the Sun
print("\n--- Solar Test ---")
inspect_best_fit_photometry(observed_sun, best_params_per, grid2, masses2, logages2, fehs2, bands=bands_full)
# Photometry inspection for young star 1
print("\n--- Young Star 1 (Pleiades HD 283518) ---")
inspect_best_fit_photometry(observed_young1, best_params_young1, grid2, masses2, logages2, fehs2, bands=bands_full)
# Photometry inspection for young star 2
print("\n--- Young Star 2 (Hyades HD 283518) ---")
inspect_best_fit_photometry(observed_young2, best_params_young2, grid2, masses2, logages2, fehs2, bands=bands_full)