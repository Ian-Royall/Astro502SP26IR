
from ezpadova import parsec 
import matplotlib.pyplot as plt # for now run with command ''QT_QPA_PLATFORM=xcb python Ezpadova_transition.py'
# this is a workaround for the "This application failed to start because no Qt platform plugin could be initialized" error on some Linux systems when using Matplotlib with certain backends.   
import numpy as np

# Single isochrone: use triplet with step=0
iso = parsec.get_isochrones(
    logage=(9.0, 9.5, 0.5),      # (start, stop, step=0) = single value
    MH=(0.0,0.0,0.0),          # same for metallicity
    photsys_file='gaiaEDR3'
)

print(iso)                       # Should now print the Astropy Table without encoding error
print("Columns:", iso.columns)

# Quick plot (G vs G - J if J exists, fallback to BP-RP)
if 'J' in iso.columns:
    color = iso['Gmag'] - iso['J']
    xlabel = 'G - J (mag)'
else:
    color = iso['G_BPmag'] - iso['G_RPmag']
    xlabel = 'G_BP - G_RP (mag)'

plt.figure(figsize=(7,8))
plt.scatter(color, iso['Gmag'], s=5, c=iso['Mini'], cmap='viridis')
plt.gca().invert_yaxis()
plt.xlabel(xlabel)
plt.ylabel('G (absolute mag)')
plt.title('PARSEC Isochrone – logAge ≈9.3 (~2 Gyr), [M/H]=0.0, Gaia EDR3')
plt.grid(alpha=0.3)
plt.colorbar(label='Initial Mass (M⊙)')
plt.show()