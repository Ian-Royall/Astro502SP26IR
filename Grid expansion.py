import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ezpadova import parsec

# Load your existing fine grid (Gaia bands only)
data = np.load('parsec_grid.npz')
grid = {k: data[k] for k in data if k not in ['masses', 'logages', 'fehs']}
masses = data['masses']
logages = data['logages']
fehs = data['fehs']

print("Loaded existing grid with bands:", list(grid.keys()))

# NIR bands to add
nir_bands = ['J', 'H', 'K', 'W1', 'W2', 'W3', 'W4']
nir_col_map = {
    'J': 'Jmag',
    'H': 'Hmag',
    'K': 'Ksmag',
    'W1': 'W1mag',
    'W2': 'W2mag',
    'W3': 'W3mag',
    'W4': 'W4mag'
}

# Initialize NIR arrays
for band in nir_bands:
    grid[band] = np.full((len(fehs), len(logages), len(masses)), np.nan)

print("Starting NIR append...")

for i_f, feh in enumerate(fehs):
    for i_la, la in enumerate(logages):
        try:
            nir_table = parsec.get_isochrones(
                logage=(la, la, 0.0),
                MH=(feh, feh, 0.0),
                photsys_file='2mass_spitzer_wise'
            )
            if len(nir_table) == 0:
                print(f"Empty NIR table at [M/H]={feh:.3f}, logAge={la:.3f}")
                continue

            nir_table = nir_table.sort_values('Mini').drop_duplicates('Mini')
            mini_nir = nir_table['Mini'].values

            for band, col in nir_col_map.items():
                if col in nir_table.columns:
                    interp = interp1d(
                        mini_nir, nir_table[col].values,
                        kind='linear',
                        bounds_error=False,
                        fill_value=np.nan
                    )
                    grid[band][i_f, i_la] = interp(masses)
                else:
                    print(f"Missing NIR column '{col}' at [M/H]={feh:.3f}, logAge={la:.3f}")
        except Exception as e:
            print(f"NIR query failed at [M/H]={feh:.3f}, logAge={la:.3f}: {e}")

# Save the augmented grid (same shape, more bands)
np.savez('parsec_grid_with_NIR.npz', **grid, masses=masses, logages=logages, fehs=fehs)
print("Augmented grid saved! Now has Gaia + NIR bands.")
print("New bands added:", [b for b in grid if b not in ['G', 'BP', 'RP']])