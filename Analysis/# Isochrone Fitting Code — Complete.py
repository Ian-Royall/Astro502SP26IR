# Isochrone Fitting Code — Complete
import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize

_ANALYSIS = os.path.dirname(os.path.abspath(__file__))
_BASE     = os.path.dirname(_ANALYSIS)

# ──────────────────────────────────────────
# Build interpolators ONCE from the 3D grid
# ──────────────────────────────────────────
def build_interpolators(grid, fehs, logages, masses):
    """
    Pre-build one RegularGridInterpolator per band.
    Call once at startup — never inside the fitting loop.
    """
    return {
        band: RegularGridInterpolator(
            (fehs, logages, masses),
            grid[band],
            bounds_error=False,
            fill_value=np.nan,
            method='linear',
        )
        for band in grid
    }

# ──────────────────────────────────────────
# get_model_mag — uses pre-built interpolators
# ──────────────────────────────────────────
def get_model_mag(mass, age_yr, feh, interpolators):
    """
    Interpolate magnitudes from pre-built interpolators.
    age_yr: age in years (e.g. 2e9) — converted to logage internally.
    """
    logage = np.log10(age_yr)
    point  = [[feh, logage, mass]]
    mags   = {}
    for band, interp in interpolators.items():
        try:
            val = float(np.asarray(interp(point)).item())
        except Exception:
            val = np.nan
        mags[band] = val
    return mags

# ──────────────────────────────────────────
# Load saved grid
# ──────────────────────────────────────────
data2 = np.load(os.path.join(_BASE, 'Grids', 'Tera Grid.npz'))

# Separate physical grids from magnitude grids
_reserved = ['masses', 'logages', 'fehs', 'Teff', 'logg']
grid2    = {k: data2[k] for k in data2 if k not in _reserved}
masses2  = data2['masses']
logages2 = data2['logages']
fehs2    = data2['fehs']

# Teff and logg grids — present only after grid is regenerated with amendments.
# Gracefully degrade if not yet present (old grid files).
teff2 = data2['Teff'] if 'Teff' in data2 else None
logg2 = data2['logg'] if 'logg' in data2 else None


def _fill_edge_nans(arr):
    """
    Fill edge NaN values along the mass axis (axis 2) by extending the
    nearest valid value outward.  All NaN in the grid are confirmed to be
    edge-only (no interior NaN exist), so forward/backward fill is safe and
    physically reasonable: querying a mass just outside the isochrone's
    coverage returns the closest valid point rather than NaN.
    """
    out = arr.copy()
    n_f, n_la, n_m = out.shape
    for i_f in range(n_f):
        for i_la in range(n_la):
            row = out[i_f, i_la, :]
            if np.all(np.isnan(row)):
                continue
            valid = np.where(np.isfinite(row))[0]
            # Forward-fill leading NaN
            if valid[0] > 0:
                out[i_f, i_la, :valid[0]] = row[valid[0]]
            # Backward-fill trailing NaN
            if valid[-1] < n_m - 1:
                out[i_f, i_la, valid[-1]+1:] = row[valid[-1]]
    return out


# Fill edge NaN before building interpolators so RegularGridInterpolator
# never encounters NaN for in-range queries.
grid2  = {k: _fill_edge_nans(v) for k, v in grid2.items()}
if teff2 is not None:
    teff2 = _fill_edge_nans(teff2)
if logg2 is not None:
    logg2 = _fill_edge_nans(logg2)

# Build magnitude interpolators once
interp2 = build_interpolators(grid2, fehs2, logages2, masses2)

# Build Teff/logg interpolators if grids are available
teff_interp = RegularGridInterpolator(
    (fehs2, logages2, masses2), teff2,
    bounds_error=False, fill_value=np.nan, method='linear'
) if teff2 is not None else None

logg_interp = RegularGridInterpolator(
    (fehs2, logages2, masses2), logg2,
    bounds_error=False, fill_value=np.nan, method='linear'
) if logg2 is not None else None

# ──────────────────────────────────────────
# Brute-force grid search
# ──────────────────────────────────────────
def brute_force_best_fit(observed, errors, grid, masses, logages, fehs, bands=None,
                         feh_bounds=None):
    """
    Fast initial grid search to find a good starting point.
    Returns best (mass, age_yr, feh) and chi².

    feh_bounds : (lo, hi) tuple to restrict the [Fe/H] search range, e.g.
                 (st_met - FEH_TOLERANCE, st_met + FEH_TOLERANCE).
                 Defaults to None (search full grid).
    """
    if bands is None:
        bands = list(observed.keys())

    best_chi2   = np.inf
    best_params = (np.nan, np.nan, np.nan)

    for i_f in range(len(fehs)):
        if feh_bounds is not None and not (feh_bounds[0] <= fehs[i_f] <= feh_bounds[1]):
            continue
        for i_la in range(len(logages)):
            for i_m in range(len(masses)):
                chi2    = 0.0
                n_valid = 0
                for band in bands:
                    if band in grid:
                        val = grid[band][i_f, i_la, i_m]
                        if not np.isnan(val):
                            chi2   += ((observed[band] - val) / errors[band]) ** 2
                            n_valid += 1
                if n_valid >= 4 and chi2 < best_chi2:
                    best_chi2   = chi2
                    best_params = (masses[i_m], 10 ** logages[i_la], fehs[i_f])

    return best_params, best_chi2

# ──────────────────────────────────────────
# Continuous fit (single start)
# ──────────────────────────────────────────
def continuous_fit(observed, errors, rough_params, interpolators,
                   bands=None, method='L-BFGS-B',
                   obs_teff=None, err_teff=100.0,
                   obs_logg=None, err_logg=0.1,
                   feh_bounds=(-2.0, 0.6)):
    """
    Continuous optimisation from rough_params = (mass, age_yr, feh).

    Optional spectroscopic priors:
        obs_teff : observed effective temperature in Kelvin
        err_teff : 1-sigma uncertainty on Teff (default 100 K)
        obs_logg : observed surface gravity (log g, cgs)
        err_logg : 1-sigma uncertainty on logg (default 0.1 dex)

    If obs_teff / obs_logg are None the prior is simply skipped,
    so old call sites without those arguments still work.

    feh_bounds : (lo, hi) tuple to restrict the [Fe/H] search range, e.g.
                 (st_met - FEH_TOLERANCE, st_met + FEH_TOLERANCE).
    """
    if bands is None:
        bands = list(observed.keys())

    bounds = [
        (0.08, 3.0),   # mass  [M⊙]
        (7.5,  10.2),  # logage (~30 Myr – 15 Gyr)
        feh_bounds,    # [Fe/H] — narrowed from table metallicity when available
    ]

    def chi2_func(params):
        mass, logage, feh = params

        # Hard boundary enforcement — catches Nelder-Mead if ever used
        if not (bounds[0][0] <= mass   <= bounds[0][1] and
                bounds[1][0] <= logage <= bounds[1][1] and
                bounds[2][0] <= feh    <= bounds[2][1]):
            return np.inf

        point = [[feh, logage, mass]]

        # ── Photometric chi² ────────────────────────────────────────────
        mags    = get_model_mag(mass, 10 ** logage, feh, interpolators)
        chi2    = 0.0
        n_valid = 0
        for band in bands:
            if band in mags and not np.isnan(mags[band]):
                chi2   += ((observed[band] - mags[band]) / errors[band]) ** 2
                n_valid += 1
        if n_valid < 3:
            return np.inf
        chi2 /= n_valid

        # ── Teff prior ──────────────────────────────────────────────────
        if obs_teff is not None and teff_interp is not None:
            model_teff = float(np.asarray(teff_interp(point)).item())
            if not np.isnan(model_teff):
                chi2 += ((obs_teff - model_teff) / err_teff) ** 2

        # ── logg prior ──────────────────────────────────────────────────
        if obs_logg is not None and logg_interp is not None:
            model_logg = float(np.asarray(logg_interp(point)).item())
            if not np.isnan(model_logg):
                chi2 += ((obs_logg - model_logg) / err_logg) ** 2

        return chi2

    # FIX: rough_params = (mass, age_yr, feh) — convert age_yr → logage
    rough_mass, rough_age_yr, rough_feh = rough_params
    initial_guess = np.clip(
        [rough_mass, np.log10(rough_age_yr), rough_feh],
        [b[0] for b in bounds],
        [b[1] for b in bounds],
    )

    result = minimize(
        chi2_func,
        initial_guess,
        bounds=bounds,
        method=method,
        options={'maxiter': 400, 'ftol': 1e-6, 'gtol': 1e-5, 'disp': False},
    )

    if result.success:
        best_mass, best_logage, best_feh = result.x
        best_age_yr = 10 ** best_logage
        print("Continuous fit success!")
        print(f"  Mass:   {best_mass:.3f} M⊙")
        print(f"  Age:    {best_age_yr / 1e9:.3f} Gyr")
        print(f"  [Fe/H]: {best_feh:.3f}")
        print(f"  χ²:     {result.fun:.4f}")
        print(f"  Message: {result.message}")
        return (best_mass, best_age_yr, best_feh), result.fun
    else:
        print("Optimisation failed:", result.message)
        return rough_params, chi2_func(initial_guess)

# ──────────────────────────────────────────
# Continuous fit with random restarts
# ──────────────────────────────────────────
def continuous_fit_with_perturbations(observed, errors, rough_params, interpolators,
                                      bands=None, n_starts=8, method='L-BFGS-B',
                                      obs_teff=None, err_teff=100.0,
                                      obs_logg=None, err_logg=0.1,
                                      feh_bounds=(-2.0, 0.6)):
    """
    Multi-start version of continuous_fit to escape local minima.
    Accepts the same optional Teff/logg priors.

    feh_bounds : (lo, hi) tuple to restrict the [Fe/H] search range, e.g.
                 (st_met - FEH_TOLERANCE, st_met + FEH_TOLERANCE).
    """
    if bands is None:
        bands = list(observed.keys())

    rough_mass, rough_age_yr, rough_feh = rough_params
    rough_logage = np.log10(rough_age_yr)

    base_bounds = [
        (max(0.08, rough_mass   * 0.7), min(3.0,  rough_mass   * 1.3)),
        (max(7.5,  rough_logage - 0.8), min(10.2, rough_logage + 0.8)),
        (max(feh_bounds[0], rough_feh - 0.8), min(feh_bounds[1], rough_feh + 0.8)),
    ]

    def chi2_func(params):
        mass, logage, feh = params

        if not (base_bounds[0][0] <= mass   <= base_bounds[0][1] and
                base_bounds[1][0] <= logage <= base_bounds[1][1] and
                base_bounds[2][0] <= feh    <= base_bounds[2][1]):
            return np.inf

        point = [[feh, logage, mass]]

        mags    = get_model_mag(mass, 10 ** logage, feh, interpolators)
        chi2    = 0.0
        n_valid = 0
        for band in bands:
            if band in mags and not np.isnan(mags[band]):
                chi2   += ((observed[band] - mags[band]) / errors[band]) ** 2
                n_valid += 1
        if n_valid < 3:
            return np.inf
        chi2 /= n_valid

        if obs_teff is not None and teff_interp is not None:
            model_teff = float(np.asarray(teff_interp(point)).item())
            if not np.isnan(model_teff):
                chi2 += ((obs_teff - model_teff) / err_teff) ** 2

        if obs_logg is not None and logg_interp is not None:
            model_logg = float(np.asarray(logg_interp(point)).item())
            if not np.isnan(model_logg):
                chi2 += ((obs_logg - model_logg) / err_logg) ** 2

        return chi2

    best_chi2   = np.inf
    best_result = None
    last_perturbed = np.array([rough_mass, rough_logage, rough_feh])

    for start in range(n_starts):
        perturbation   = np.random.uniform(-0.15, 0.15, 3)
        perturbed      = np.array([rough_mass, rough_logage, rough_feh]) * (1 + perturbation)
        perturbed      = np.clip(perturbed,
                                 [b[0] for b in base_bounds],
                                 [b[1] for b in base_bounds])
        last_perturbed = perturbed

        print(f"Start {start + 1}/{n_starts}: initial = {perturbed}")

        result = minimize(
            chi2_func,
            perturbed,
            bounds=base_bounds,
            method=method,
            options={'maxiter': 400, 'ftol': 1e-6, 'gtol': 1e-5, 'disp': False},
        )

        if result.fun < best_chi2:
            best_chi2   = result.fun
            best_result = result

    if best_result is not None and best_result.success:
        best_mass, best_logage, best_feh = best_result.x
        best_age_yr = 10 ** best_logage
        print("Best continuous fit across starts:")
        print(f"  Mass:   {best_mass:.3f} M⊙")
        print(f"  Age:    {best_age_yr / 1e9:.3f} Gyr")
        print(f"  [Fe/H]: {best_feh:.3f}")
        print(f"  χ²:     {best_chi2:.4f}")
        print(f"  Message: {best_result.message}")
        return (best_mass, best_age_yr, best_feh), best_chi2
    else:
        print("All starts failed or did not converge")
        return tuple(last_perturbed), chi2_func(last_perturbed)

# ──────────────────────────────────────────
# Photometry inspection helper
# ──────────────────────────────────────────
def inspect_best_fit_photometry(observed, errors, params, interpolators,
                                bands, label=""):
    """
    Print a band-by-band comparison of observed vs model magnitudes.
    Now takes errors as an argument so it doesn't rely on global error dicts.
    """
    mass, age_yr, feh = params
    model_mags = get_model_mag(mass, age_yr, feh, interpolators)

    print(f"\n=== {label} ===")
    print(f"Parameters: mass={mass:.3f} M⊙, age={age_yr/1e6:.0f} Myr, [Fe/H]={feh:.3f}")
    print("Band   Obs     Model    Residual    χ² term")
    print("---------------------------------------------")
    total_chi2 = 0.0
    for band in bands:
        if band in observed and band in model_mags:
            obs = observed[band]
            mod = model_mags[band]
            err = errors.get(band, 0.03)
            if np.isnan(mod):
                print(f"{band:4}   {obs:.3f}     NaN       —")
                continue
            res       = obs - mod
            chi_term  = (res / err) ** 2
            total_chi2 += chi_term
            print(f"{band:4}   {obs:.3f}   {mod:.3f}   {res:+7.3f}    {chi_term:.2f}")
    print(f"Total χ²: {total_chi2:.2f}\n")

# ──────────────────────────────────────────
# Distance correction helper
# ──────────────────────────────────────────
def apparent_to_absolute(mags, distance_pc):
    """Convert apparent magnitude dict to absolute magnitudes."""
    mu = 5 * np.log10(distance_pc / 10)
    return {band: mag - mu for band, mag in mags.items()}

# ══════════════════════════════════════════
# TEST CASES
# ══════════════════════════════════════════

# ── Solar test ───────────────────────────
# Solar absolute magnitudes are already absolute (Sun at 10 pc by definition)
observed_sun = {
    'G':  4.65,
    'BP': 4.83,
    'RP': 4.24,
    'J':  3.64,
    'H':  3.32,
    'K':  3.28,
    'W1': 3.25,
    'W2': 3.24,
    'W3': 3.23,
    'W4': 3.22,
}
errors_sun = {
    'G':  0.01,
    'BP': (0.02**2 + 0.01**2)**0.5,
    'RP': (0.02**2 + 0.01**2)**0.5,
    'J':  0.03,
    'H':  0.03,
    'K':  0.03,
    'W1': 0.03,
    'W2': 0.03,
    'W3': 0.03,
    'W4': 0.03,
}
# Known solar spectroscopic parameters
sun_teff  = 5778.0  # K
sun_logg  = 4.44    # dex
sun_st_met = 0.00   # [Fe/H]

# ── Pleiades test star: HII 1132 (~0.9 Msun K dwarf) ────────────────────────
# Apparent magnitudes from Gaia DR3 + 2MASS — distance 136 pc
observed_pleiades_apparent = {
    'G':  10.847,
    'BP': 11.474,
    'RP': 10.077,
    'J':  9.409,
    'H':  8.972,
    'K':  8.852,
    'W1': np.nan,   # fill in from ALLWISE
    'W2': np.nan,
    'W3': np.nan,
    'W4': np.nan,
}
d_pleiades = 136.0  # pc
observed_pleiades = apparent_to_absolute(observed_pleiades_apparent, d_pleiades)
errors_pleiades = {b: 0.03 for b in observed_pleiades}
errors_pleiades['G']  = 0.01
errors_pleiades['BP'] = (0.02**2 + 0.01**2)**0.5
errors_pleiades['RP'] = (0.02**2 + 0.01**2)**0.5
# Approximate spectroscopic params for a ~K2 dwarf
pleiades_teff   = 4900.0
pleiades_logg   = 4.55
pleiades_st_met = 0.03   # Pleiades cluster [Fe/H] (roughly solar)

# ── Hyades test star: HD 28099 (~1.0 Msun G dwarf) ──────────────────────────
# Apparent magnitudes from Gaia DR3 + 2MASS — distance 47 pc
observed_hyades_apparent = {
    'G':  8.068,
    'BP': 8.543,
    'RP': 7.449,
    'J':  6.777,
    'H':  6.476,
    'K':  6.394,
    'W1': np.nan,   # fill in from ALLWISE
    'W2': np.nan,
    'W3': np.nan,
    'W4': np.nan,
}
d_hyades = 47.0  # pc
observed_hyades = apparent_to_absolute(observed_hyades_apparent, d_hyades)
errors_hyades = {b: 0.03 for b in observed_hyades}
errors_hyades['G']  = 0.01
errors_hyades['BP'] = (0.02**2 + 0.01**2)**0.5
errors_hyades['RP'] = (0.02**2 + 0.01**2)**0.5
# Approximate spectroscopic params for a ~G8 dwarf
hyades_teff   = 5500.0
hyades_logg   = 4.40
hyades_st_met = 0.13   # Hyades cluster [Fe/H]


# ══════════════════════════════════════════
# RUN FITS
# ══════════════════════════════════════════

# ── Metallicity tolerance — half-width around table [Fe/H] used as fit bounds ──
FEH_TOLERANCE = 0.15  # dex

def feh_window(st_met, tol=FEH_TOLERANCE):
    """Return (lo, hi) [Fe/H] bounds centred on the table metallicity."""
    return (st_met - tol, st_met + tol)

# ── Band selection — comment out any line here to drop that band from all fits ─
ACTIVE_BANDS = [
    'G',
    'BP',
    'RP',
    'J',
    'H',
    'K',
    'W1',
    'W2',
    'W3',
    'W4',
]

def _bands_for(obs):
    """ACTIVE_BANDS filtered to bands that are measured (non-NaN) and in the grid."""
    return [b for b in ACTIVE_BANDS if b in obs and b in grid2 and np.isfinite(obs[b])]

# ── Spectroscopic priors — comment out either line to disable that prior globally ─
ACTIVE_PRIORS = [
     'teff',
   #  'logg',
]

def _prior(val, key):
    """Return val if key is in ACTIVE_PRIORS, else None (prior skipped in fit)."""
    return val if key in ACTIVE_PRIORS else None

bands_sun      = _bands_for(observed_sun)
bands_pleiades = _bands_for(observed_pleiades)
bands_hyades   = _bands_for(observed_hyades)
print("Active bands:  ", ACTIVE_BANDS)
print("Active priors: ", ACTIVE_PRIORS)

# ── Solar fit ────────────────────────────
print("\n" + "="*50)
print("SOLAR FIT")
print("="*50)
rough_params_sun, rough_chi2_sun = brute_force_best_fit(
    observed_sun, errors_sun, grid2, masses2, logages2, fehs2,
    bands=bands_sun, feh_bounds=feh_window(sun_st_met),
)
print(f"Rough fit: {rough_params_sun}, χ²={rough_chi2_sun:.3f}")

best_params_sun, best_chi2_sun = continuous_fit(
    observed_sun, errors_sun,
    rough_params_sun, interp2,
    bands=bands_sun,
    obs_teff=_prior(sun_teff, 'teff'), err_teff=100.0,
    obs_logg=_prior(sun_logg, 'logg'), err_logg=0.1,
    feh_bounds=feh_window(sun_st_met),
)

# ── Pleiades fit ─────────────────────────
print("\n" + "="*50)
print("PLEIADES FIT (HII 1132, d=136 pc)")
print("="*50)
rough_params_pl, rough_chi2_pl = brute_force_best_fit(
    observed_pleiades, errors_pleiades, grid2, masses2, logages2, fehs2,
    bands=bands_pleiades, feh_bounds=feh_window(pleiades_st_met),
)
print(f"Rough fit: {rough_params_pl}, χ²={rough_chi2_pl:.3f}")

best_params_pl, best_chi2_pl = continuous_fit(
    observed_pleiades, errors_pleiades,
    rough_params_pl, interp2,
    bands=bands_pleiades,
    obs_teff=_prior(pleiades_teff, 'teff'), err_teff=150.0,
    obs_logg=_prior(pleiades_logg, 'logg'), err_logg=0.15,
    feh_bounds=feh_window(pleiades_st_met),
)

# ── Hyades fit ───────────────────────────
print("\n" + "="*50)
print("HYADES FIT (HD 28099, d=47 pc)")
print("="*50)
rough_params_hy, rough_chi2_hy = brute_force_best_fit(
    observed_hyades, errors_hyades, grid2, masses2, logages2, fehs2,
    bands=bands_hyades, feh_bounds=feh_window(hyades_st_met),
)
print(f"Rough fit: {rough_params_hy}, χ²={rough_chi2_hy:.3f}")

best_params_hy, best_chi2_hy = continuous_fit(
    observed_hyades, errors_hyades,
    rough_params_hy, interp2,
    bands=bands_hyades,
    obs_teff=_prior(hyades_teff, 'teff'), err_teff=150.0,
    obs_logg=_prior(hyades_logg, 'logg'), err_logg=0.15,
    feh_bounds=feh_window(hyades_st_met),
)


# ══════════════════════════════════════════
# PHOTOMETRY INSPECTION
# ══════════════════════════════════════════

inspect_best_fit_photometry( # Sun fit
    observed_sun, errors_sun, best_params_sun, interp2,
    bands=bands_sun, label="Solar fit"
)
inspect_best_fit_photometry( # Pleiades fit
    observed_pleiades, errors_pleiades, best_params_pl, interp2,
    bands=bands_pleiades, label="Pleiades fit (HII 1132)"
)
inspect_best_fit_photometry( # Hyades fit
    observed_hyades, errors_hyades, best_params_hy, interp2,
    bands=bands_hyades, label="Hyades fit (HD 28099)"
)


# ══════════════════════════════════════════
# GRID SELF-CONSISTENCY TEST
# Tests that the fitter can recover parameters from the grid's own magnitudes.
# chi² should be near zero and parameters should match input closely.
# ══════════════════════════════════════════
print("\n" + "="*50)
print("GRID SELF-CONSISTENCY TEST")
print("="*50)

test_mass   = 1.0
test_age_yr = 4.6e9
test_feh    = 0.0

synthetic = get_model_mag(test_mass, test_age_yr, test_feh, interp2)
synthetic = {b: v for b, v in synthetic.items() if not np.isnan(v)}
syn_errors = {b: 0.01 for b in synthetic}

rough_syn, rough_syn_chi2 = brute_force_best_fit(
    synthetic, syn_errors, grid2, masses2, logages2, fehs2,
    bands=list(synthetic.keys())
)
print(f"Rough: {rough_syn}, χ²={rough_syn_chi2:.6f}")

best_syn, best_syn_chi2 = continuous_fit(
    synthetic, syn_errors,
    rough_syn, interp2,
    bands=list(synthetic.keys()),
)
print(f"Recovered: mass={best_syn[0]:.4f}, age={best_syn[1]/1e9:.4f} Gyr, "
      f"feh={best_syn[2]:.4f}, χ²={best_syn_chi2:.6f}")
print(f"True:      mass={test_mass:.4f}, age={test_age_yr/1e9:.4f} Gyr, "
      f"feh={test_feh:.4f}")
print(f"Residuals: Δmass={best_syn[0]-test_mass:+.4f}, "
      f"Δage={( best_syn[1]-test_age_yr)/1e9:+.4f} Gyr, "
      f"Δfeh={best_syn[2]-test_feh:+.4f}")


# ══════════════════════════════════════════
# GRID DIAGNOSTIC — NaN distribution
# ══════════════════════════════════════════
print("\n" + "="*50)
print("NaN DIAGNOSTIC")
print("="*50)
for band in ['G', 'J']:
    print(f"\nNaN vs logage — band: {band}")
    for i_la, la in enumerate(logages2):
        n_nan = int(np.sum(np.isnan(grid2[band][:, i_la, :])))
        total = grid2[band][:, i_la, :].size
        pct   = 100 * n_nan / total
        print(f"  logage={la:.2f}: {n_nan}/{total} NaNs ({pct:.1f}%)")


# ══════════════════════════════════════════
# CSV BATCH FIT — all targets
# ══════════════════════════════════════════

PHOT_CSV    = os.path.join(_ANALYSIS, 'ASTR502_Master_Photometry_List.csv')
TARGET_CSV  = os.path.join(_ANALYSIS, 'ASTR502_Mega_Target_List.csv')
RESULTS_CSV = os.path.join(_BASE, 'Outputs', 'stellar_ages_results.csv')

# band → (apparent-mag column, error column, minimum error floor)
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


def run_csv_batch_fit():
    phot_df   = pd.read_csv(PHOT_CSV)
    target_df = pd.read_csv(TARGET_CSV)

    # Stellar params are star-level — collapse one row per hostname
    phot_df   = phot_df.drop_duplicates('hostname', keep='first').reset_index(drop=True)
    target_df = target_df.drop_duplicates('hostname', keep='first')

    merged = phot_df.merge(
        target_df[['hostname', 'bj_dist_pc', 'st_met', 'st_teff', 'st_logg']],
        on='hostname', how='inner',
    ).reset_index(drop=True)

    print(f"Batch fit: {len(merged)} unique stars")
    results = []

    for idx, row in merged.iterrows():
        hostname = row['hostname']
        dist_pc  = row['bj_dist_pc']
        st_met   = row['st_met']
        st_teff  = row['st_teff']
        st_logg  = row['st_logg']

        if not np.isfinite(dist_pc):
            print(f"  Skipping {hostname}: no distance")
            continue
        if not np.isfinite(st_met):
            print(f"  Skipping {hostname}: no metallicity")
            continue

        mu = 5 * np.log10(dist_pc / 10)

        # Build observed absolute mags and errors
        observed = {}
        errors   = {}
        for band, (mag_col, err_col, min_err) in BAND_COLS.items():
            if mag_col not in row.index:
                continue
            app_mag = row[mag_col]
            if not np.isfinite(app_mag):
                continue
            observed[band] = app_mag - mu
            raw_err = row[err_col] if err_col in row.index else np.nan
            errors[band] = max(
                float(raw_err) if np.isfinite(raw_err) else min_err,
                min_err,
            )

        bands = _bands_for(observed)
        if len(bands) < 3:
            print(f"  Skipping {hostname}: only {len(bands)} usable band(s)")
            continue

        fb       = feh_window(st_met)
        teff_arg = _prior(float(st_teff) if np.isfinite(st_teff) else None, 'teff')
        logg_arg = _prior(float(st_logg) if np.isfinite(st_logg) else None, 'logg')

        print(f"\n{'='*50}")
        print(f"[{idx + 1}/{len(merged)}] {hostname}  "
              f"d={dist_pc:.1f} pc  [Fe/H]={st_met:.3f}  bands={bands}")
        print(f"{'='*50}")

        try:
            rough, rough_chi2 = brute_force_best_fit(
                observed, errors, grid2, masses2, logages2, fehs2,
                bands=bands, feh_bounds=fb,
            )
            print(f"  Rough: mass={rough[0]:.3f}, age={rough[1]/1e9:.3f} Gyr, "
                  f"feh={rough[2]:.3f}, χ²={rough_chi2:.3f}")

            best, best_chi2 = continuous_fit(
                observed, errors, rough, interp2,
                bands=bands,
                obs_teff=teff_arg, err_teff=100.0,
                obs_logg=logg_arg, err_logg=0.1,
                feh_bounds=fb,
            )
            age_gyr_fit = best[1] / 1e9
            age_flag = (
                'lower_boundary' if age_gyr_fit <= 10 ** logages2[0]  / 1e9 * 1.001 else
                'upper_boundary' if age_gyr_fit >= 10 ** logages2[-1] / 1e9 * 0.999 else
                'ok'
            )
            chi2_flag = (
                'inf'  if not np.isfinite(best_chi2) else
                'high' if best_chi2 > 10             else
                'ok'
            )
            results.append({
                'hostname':   hostname,
                'mass_msun':  round(best[0], 4),
                'age_gyr':    round(age_gyr_fit, 4),
                'feh':        round(best[2], 4),
                'chi2':       round(best_chi2, 4) if np.isfinite(best_chi2) else np.inf,
                'n_bands':    len(bands),
                'dist_pc':    round(dist_pc, 2),
                'st_met_in':  round(st_met, 4),
                'age_flag':   age_flag,
                'chi2_flag':  chi2_flag,
                'reliable':   age_flag == 'ok' and chi2_flag == 'ok',
            })

        except Exception as e:
            print(f"  Fit failed for {hostname}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\nDone — {len(results_df)} stars fitted, results saved to '{RESULTS_CSV}'.")

    # Quality summary
    n_reliable = results_df['reliable'].sum()
    n_lower    = (results_df['age_flag'] == 'lower_boundary').sum()
    n_upper    = (results_df['age_flag'] == 'upper_boundary').sum()
    n_highchi2 = (results_df['chi2_flag'] == 'high').sum()
    n_inf      = (results_df['chi2_flag'] == 'inf').sum()
    print(f"\nQuality summary:")
    print(f"  Reliable (age ok + chi2 <= 10): {n_reliable} ({100*n_reliable/len(results_df):.1f}%)")
    print(f"  Age pinned — lower boundary:    {n_lower}")
    print(f"  Age pinned — upper boundary:    {n_upper}")
    print(f"  High chi2 (>10, not boundary):  {n_highchi2}")
    print(f"  Inf chi2 (optimizer failed):    {n_inf}")

    return results_df

results_df = run_csv_batch_fit()