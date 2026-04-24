import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_CSV = '/home/iroyall/Documents/Astro 502/Astro502SP26IR/stellar_ages_results.csv'

df  = pd.read_csv(RESULTS_CSV)
rel = df[df['reliable'] == True].copy()

# ── Age Distribution Bar Chart ────────────────────────────────────────────────
bins   = np.arange(0, 10.25, 0.25)
labels = [f'{b:.2f}' for b in bins[:-1]]
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(labels)))

counts, _ = np.histogram(rel['age_gyr'], bins=bins)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=0.6)

# Value labels on top of each bar
for bar, count in zip(bars, counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                str(count), ha='center', va='bottom', fontsize=9, color='0.3')

ax.set_xlabel('Stellar Age (Gyr)', fontsize=12)
ax.set_ylabel('Number of Stars', fontsize=12)
ax.set_title(f'Age Distribution of Reliable Exoplanet Host Stars Tera (with t_eff) (n = {len(rel)})',
             fontsize=13, pad=12)
ax.set_ylim(0, counts.max() * 1.15)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([f'{b:.2f}' if i % 4 == 0 else '' for i, b in enumerate(bins[:-1])],
                   rotation=45, ha='right')
ax.tick_params(axis='x', labelsize=8)

plt.tight_layout()
plt.savefig('/home/iroyall/Documents/Astro 502/Astro502SP26IR/age_distribution_tera.png', dpi=150)
plt.show()
print("Saved: age_distribution_tera.png")


# ── Fitted Age vs Catalog Age Scatter ─────────────────────────────────────────
target  = pd.read_csv('/home/iroyall/Documents/Astro 502/Astro502SP26IR/ASTR502_Mega_Target_List.csv')
target  = target.drop_duplicates('hostname', keep='first')

comp = df.merge(target[['hostname', 'st_age', 'st_ageerr1', 'st_ageerr2']],
                on='hostname', how='inner')
comp = comp[np.isfinite(comp['st_age']) & np.isfinite(comp['age_gyr'])].copy()

# Reliable stars only, excluding the untrustworthy <0.25 Gyr bin
reliable = comp[(comp['reliable'] == True) & (comp['age_gyr'] >= 0.25)]

# Stats — ODR accounts for errors in both axes (catalog age uncertainty is large)
from scipy import stats, odr

x    = reliable['st_age'].values
y    = reliable['age_gyr'].values
xerr = ((reliable['st_ageerr1'].fillna(reliable['st_ageerr1'].median()) -
         reliable['st_ageerr2'].fillna(reliable['st_ageerr2'].median())) / 2).abs().clip(lower=0.1).values
yerr = np.full(len(y), 1.5)   # approximate fit uncertainty

odr_data  = odr.Data(x, y, wd=1/xerr**2, we=1/yerr**2)
odr_model = odr.Model(lambda B, x: B[0]*x + B[1])
odr_run   = odr.ODR(odr_data, odr_model, beta0=[1.0, 0.0]).run()
slope, intercept = odr_run.beta

_, _, r, _, _ = stats.linregress(x, y)   # keep Pearson R for reference
median_offset = (reliable['age_gyr'] - reliable['st_age']).median()

mass_min, mass_max = 0.3, 2.0

fig, ax = plt.subplots(figsize=(9, 8))

# Error bars in neutral gray so they don't clash with the color gradient
ax.errorbar(reliable['st_age'], reliable['age_gyr'],
            xerr=[-reliable['st_ageerr2'].fillna(0), reliable['st_ageerr1'].fillna(0)],
            fmt='none', ecolor='0.7', alpha=0.4, linewidth=0.6, zorder=3)

sc = ax.scatter(reliable['st_age'], reliable['age_gyr'],
                c=reliable['mass_msun'], cmap='plasma',
                vmin=mass_min, vmax=mass_max,
                s=28, linewidths=0.3, edgecolors='0.3', zorder=4,
                label=f'Reliable fits, age ≥ 0.25 Gyr (n={len(reliable)})')

cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
cbar.set_label('Stellar Mass (M☉)', fontsize=10)

# 1:1 line
lim = (0, 14)
ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.5, label='1:1')

# ODR best-fit line
x_line = np.linspace(0, 13, 100)
ax.plot(x_line, slope * x_line + intercept, 'tomato', linewidth=1.5, alpha=0.8,
        label=f'ODR fit  (slope={slope:.2f}, R={r:.2f})')

ax.text(0.04, 0.97, f'Median offset: {median_offset:+.2f} Gyr',
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

ax.set_xlim(*lim)
ax.set_ylim(0, 10.5)
ax.set_xlabel('Catalog Age (Gyr)', fontsize=12)
ax.set_ylabel('Isochrone Fit Age (Gyr)', fontsize=12)
ax.set_title('Isochrone Fit Age vs Catalog Age — Exoplanet Host Stars Tera (with t_eff)', fontsize=13, pad=12)
ax.legend(fontsize=10, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/iroyall/Documents/Astro 502/Astro502SP26IR/age_comparison_tera.png', dpi=150)
plt.show()
print("Saved: age_comparison_tera.png")


# ── My Fit Age vs James's Fit Age ─────────────────────────────────────────────
james = pd.read_csv('/home/iroyall/Documents/Astro 502/Astro502SP26IR/James_interpolate_20260414_161013_candidate_fits.csv')
james['age_gyr_james'] = james['age_yr'] / 1e9

vs_james = df.merge(james[['hostname', 'age_gyr_james', 'chi2_reduced']], on='hostname', how='inner')
vs_james = vs_james[np.isfinite(vs_james['age_gyr_james'])].copy()

# My reliable (age >= 0.25 Gyr) vs all of James's
rel_vs_james = vs_james[(vs_james['reliable'] == True) & (vs_james['age_gyr'] >= 0.25)]

fig, ax = plt.subplots(figsize=(9, 8))

sc2 = ax.scatter(rel_vs_james['age_gyr_james'], rel_vs_james['age_gyr'],
                 c=rel_vs_james['mass_msun'], cmap='plasma',
                 vmin=mass_min, vmax=mass_max,
                 s=28, linewidths=0.3, edgecolors='0.3', zorder=4,
                 label=f'My reliable fits, age ≥ 0.25 Gyr (n={len(rel_vs_james)})')

cbar2 = plt.colorbar(sc2, ax=ax, pad=0.02, fraction=0.046)
cbar2.set_label('Stellar Mass (M☉)', fontsize=10)

lim = (0, 14)
ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.5, label='1:1')
ax.set_xlim(*lim)
ax.set_ylim(0, 10.5)

ax.set_xlabel("James's Fit Age (Gyr)", fontsize=12)
ax.set_ylabel('My Isochrone Fit Age (Gyr)', fontsize=12)
ax.set_title('My Fit Age vs James\'s Fit Age — Exoplanet Host Stars', fontsize=13, pad=12)
ax.legend(fontsize=10, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/iroyall/Documents/Astro 502/Astro502SP26IR/age_vs_james.png', dpi=150)
plt.show()
print("Saved: age_vs_james.png")
