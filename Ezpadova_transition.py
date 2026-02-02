
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

# Now safe to import everything
from ezpadova import parsec
import matplotlib.pyplot as plt
import numpy as np

# Single isochrone: use triplet with step=0
iso = parsec.get_isochrones(
    logage=(9.3, 9.3, 0.0),      # (start, stop, step=0) = single value
    MH=(0.0, 0.0, 0.0),          # same for metallicity
    photsys_file='gaiaedr3'
)

print(iso)                       # Should now print the Astropy Table without encoding error
print("Columns:", iso.colnames)

# Quick plot (G vs G - J if J exists, fallback to BP-RP)
if 'J' in iso.colnames:
    color = iso['G'] - iso['J']
    xlabel = 'G - J (mag)'
else:
    color = iso['G_BP'] - iso['G_RP']
    xlabel = 'G_BP - G_RP (mag)'

plt.figure(figsize=(7,8))
plt.scatter(color, iso['G'], s=5, c=iso['Mini'], cmap='viridis')
plt.gca().invert_yaxis()
plt.xlabel(xlabel)
plt.ylabel('G (absolute mag)')
plt.title('PARSEC Isochrone – logAge ≈9.3 (~2 Gyr), [M/H]=0.0, Gaia EDR3')
plt.grid(alpha=0.3)
plt.colorbar(label='Initial Mass (M⊙)')
plt.show()