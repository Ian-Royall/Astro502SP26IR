# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 10:28:10 2026

@author: Ian
"""


# models generate magnitudes, extinction could harm extinciton. Dust maps like brutus could help observe such effects.


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np


# Load the file (skip header lines until data starts; usually after the column names line)
# Find the line with "Zini MH ..." and use that as header

df = pd.read_csv('output1_test.dat',  # 1 Gyr Isochrone Import
                 delim_whitespace=True,
                 comment='#',
                 skiprows=lambda x: x < 20,  # adjust if needed to skip header
                 names=['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe',
                        'logg', 'label', 'McoreTP', 'C_O', 'period0', 'period1', 'period2',
                        'period3', 'period4', 'pmode', 'Mloss', 'tau1m', 'X', 'Y', 'Xc',
                        'Xn', 'Xo', 'Cexcess', 'Z', 'mbolmag', 'Umag', 'Bmag', 'Vmag',
                        'Rmag', 'Imag', 'Jmag', 'Hmag', 'Kmag'])


# CMD (color-magnitude): B-V vs V (absolute mag)

# Plotting 1Gyr
df['BV'] = df['Bmag'] - df['Vmag']
plt.figure(figsize=(6, 8))
plt.scatter(df['BV'], df['Vmag'], s=5, marker='^',
            c=df['Mini'], cmap='viridis')


# Plot Settings and labels
plt.gca().invert_yaxis()  # Brighter (lower mag) on top
plt.xlabel('B - V (mag)')
plt.ylabel('V (absolute mag)')
plt.colorbar(label='Initial Mass (Msun)')
plt.title('PARSEC CMD: 1 Gyr vs 5 Gyr Isochrones, UBVRIJHK system')
plt.grid(alpha=0.3)


plt.show()
