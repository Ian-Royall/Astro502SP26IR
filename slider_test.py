import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import numpy as np

# Load the entire multi-age file
df_all = pd.read_csv(
    'output1_test.dat',           
    delim_whitespace=True,
    comment='#',
    skiprows=lambda x: x < 20,           
    names=['Zini', 'MH', 'logAge', 'Mini', 'int_IMF', 'Mass', 'logL', 'logTe',
           'logg', 'label', 'McoreTP', 'C_O', 'period0', 'period1', 'period2',
           'period3', 'period4', 'pmode', 'Mloss', 'tau1m', 'X', 'Y', 'Xc',
           'Xn', 'Xo', 'Cexcess', 'Z', 'mbolmag', 'Umag', 'Bmag', 'Vmag',
           'Rmag', 'Imag', 'Jmag', 'Hmag', 'Kmag']
)

# Compute Color
df_all['BV'] = df_all['Bmag'] - df_all['Vmag']

# Group by logAge and store each isochrone separately
grouped = df_all.groupby('logAge')
isochrones = {logage: group for logage, group in grouped}

# Convert logAge to linear age in Gyr for nicer display
# e.g. 9.0 → 1 Gyr, 9.6 → ~4 Gyr, etc.
ages_gyr = {logage: 10**(logage - 9) for logage in isochrones.keys()}

# Sorted list of logAges (for slider)
logages_sorted = sorted(isochrones.keys())
ages_display = [ages_gyr[la]
                for la in logages_sorted]  # for slider labels/values


# Create figure
fig, ax = plt.subplots(figsize=(7, 9))
plt.subplots_adjust(bottom=0.18)  # room for slider

# Start with the first (youngest) isochrone
initial_logage = logages_sorted[0]
current_df = isochrones[initial_logage]

scat = ax.scatter(
    current_df['BV'],
    current_df['Vmag'],
    s=6,
    marker='^',
    c=current_df['Mini'],
    cmap='viridis',
    alpha=0.9
)

ax.invert_yaxis()
ax.set_xlabel('B − V (mag)')
ax.set_ylabel('V (absolute mag)')
ax.grid(alpha=0.3)

cbar = fig.colorbar(scat, ax=ax, label='Initial Mass (M$_\odot$)')
ax.set_title(
    f'PARSEC CMD – logAge = {initial_logage:.2f} → Age ≈ {ages_gyr[initial_logage]:.1f} Gyr')

# Slider (using linear age in Gyr for user-friendliness)
ax_slider = plt.axes([0.15, 0.06, 0.7, 0.04])
slider = mwidgets.Slider(
    ax=ax_slider,
    label='Age (Gyr)',
    valmin=min(ages_display),
    valmax=max(ages_display),
    valinit=ages_display[0],
    # small step for smoothness
    valstep=np.diff(ages_display).min() / 2 if len(ages_display) > 1 else 0.1,
    color='teal'
)

# Update function: find closest logAge when slider moves


def update(val):
    # Find the logAge whose Gyr age is closest to the slider value
    closest_logage = min(
        logages_sorted, key=lambda la: abs(ages_gyr[la] - val))
    df = isochrones[closest_logage]

    scat.set_offsets(np.c_[df['BV'], df['Vmag']])
    scat.set_array(df['Mini'])

    ax.set_title(
        f'PARSEC CMD – logAge = {closest_logage:.2f} → Age ≈ {ages_gyr[closest_logage]:.1f} Gyr')
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()