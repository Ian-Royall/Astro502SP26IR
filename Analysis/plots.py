import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats, odr

_ANALYSIS = os.path.dirname(os.path.abspath(__file__))
_BASE     = os.path.dirname(_ANALYSIS)
_OUTPUTS  = os.path.join(_BASE, 'Outputs')

RESULTS_CSV = os.path.join(_OUTPUTS, 'stellar_ages_results.csv')
MCMC_CSV    = os.path.join(_OUTPUTS, 'mcmc_results.csv')

df   = pd.read_csv(RESULTS_CSV)
mcmc = pd.read_csv(MCMC_CSV)

# Merge batch fit + MCMC; fall back to batch age if MCMC not available
df = df.merge(mcmc, on='hostname', how='left')
df['age_plot']   = df['age_50'].fillna(df['age_gyr'])
df['age_err_lo'] = df['age_err_lo'].fillna(1.5)
df['age_err_hi'] = df['age_err_hi'].fillna(1.5)

rel = df[(df['reliable'] == True) & (df['age_gyr'] >= 0.25)].copy()

# Exclude posteriors where the sampler never mixed
if 'accept_fr' in rel.columns:
    rel = rel[rel['accept_fr'].fillna(0.5) >= 0.15]

mass_min, mass_max = 0.3, 2.0

print(f"Reliable stars for plots: {len(rel)}")
print(f"  Median MCMC age uncertainty: -{rel['age_err_lo'].median():.2f} / +{rel['age_err_hi'].median():.2f} Gyr")


# ── 1. Age Distribution Bar Chart ─────────────────────────────────────────────
bins   = np.arange(0, 10.25, 0.25)
labels = [f'{b:.2f}' for b in bins[:-1]]
colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(labels)))

counts, _ = np.histogram(rel['age_plot'], bins=bins)

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(labels, counts, color=colors, edgecolor='white', linewidth=0.6)
for bar, count in zip(bars, counts):
    if count > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                str(count), ha='center', va='bottom', fontsize=9, color='0.3')

ax.set_xlabel('Stellar Age (Gyr)', fontsize=12)
ax.set_ylabel('Number of Stars', fontsize=12)
ax.set_title(f'Age Distribution of Reliable Exoplanet Host Stars  (n = {len(rel)})',
             fontsize=13, pad=12)
ax.set_ylim(0, counts.max() * 1.15)
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels([f'{b:.2f}' if i % 4 == 0 else '' for i, b in enumerate(bins[:-1])],
                   rotation=45, ha='right')
ax.tick_params(axis='x', labelsize=8)

# Mean ± std annotation
mu_age = rel['age_plot'].mean()
sd_age = rel['age_plot'].std()
ax.axvline(mu_age / 0.25, color='steelblue', linewidth=1.4, linestyle='--', alpha=0.8)
ax.text(0.97, 0.95, f'Mean = {mu_age:.2f} Gyr\nStd  = {sd_age:.2f} Gyr',
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.85))

plt.tight_layout()
plt.savefig(os.path.join(_OUTPUTS, 'age_distribution.png'), dpi=150)
plt.close()
print("Saved: age_distribution.png")


# ── 2. Fitted Age (MCMC) vs Catalog Age ───────────────────────────────────────
target = pd.read_csv(os.path.join(_ANALYSIS, 'ASTR502_Mega_Target_List.csv'))
target = target.drop_duplicates('hostname', keep='first')

comp = rel.merge(target[['hostname', 'st_age', 'st_ageerr1', 'st_ageerr2']],
                 on='hostname', how='inner')
comp = comp[np.isfinite(comp['st_age'])].copy()

x    = comp['st_age'].values
y    = comp['age_plot'].values
xerr = ((comp['st_ageerr1'].fillna(comp['st_ageerr1'].median()).abs() +
         comp['st_ageerr2'].fillna(comp['st_ageerr2'].median()).abs()) / 2).clip(lower=0.1).values
yerr = ((comp['age_err_lo'] + comp['age_err_hi']) / 2).values

# Weighted ODR — down-weights poorly constrained MCMC posteriors
odr_data  = odr.Data(x, y, wd=1/xerr**2, we=1/yerr**2)
odr_run   = odr.ODR(odr_data, odr.Model(lambda B, x: B[0]*x + B[1]), beta0=[1.0, 0.0]).run()
slope, intercept = odr_run.beta
_, _, r, _, _    = stats.linregress(x, y)

resid      = y - x
bias       = resid.mean()
resid_std  = resid.std()
nmad       = 1.4826 * np.median(np.abs(resid - np.median(resid)))
w          = 1.0 / yerr**2
w_resid    = np.average(resid, weights=w)

print(f"\nCatalog comparison ({len(comp)} stars):")
print(f"  Bias (mean residual):       {bias:+.2f} Gyr")
print(f"  Residual std:               {resid_std:.2f} Gyr")
print(f"  NMAD (robust scatter):      {nmad:.2f} Gyr")
print(f"  Weighted mean residual:     {w_resid:+.2f} Gyr")
print(f"  ODR slope / intercept:      {slope:.3f} / {intercept:.3f}")
print(f"  Pearson R:                  {r:.3f}")
print(f"  Median MCMC age err:        -{comp['age_err_lo'].median():.2f} / +{comp['age_err_hi'].median():.2f} Gyr")

fig, ax = plt.subplots(figsize=(9, 8))

ax.errorbar(x, y,
            xerr=[-comp['st_ageerr2'].fillna(0).values, comp['st_ageerr1'].fillna(0).values],
            yerr=[comp['age_err_lo'].values, comp['age_err_hi'].values],
            fmt='none', ecolor='0.65', alpha=0.35, linewidth=0.6, zorder=3)

sc = ax.scatter(x, y,
                c=comp['mass_msun'], cmap='plasma',
                vmin=mass_min, vmax=mass_max,
                s=28, linewidths=0.3, edgecolors='0.3', zorder=4,
                label=f'Reliable fits, age ≥ 0.25 Gyr  (n={len(comp)})')

cbar = plt.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
cbar.set_label('Stellar Mass (M☉)', fontsize=10)

lim = (0, 14)
ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.5, label='1:1')
x_line = np.linspace(0, 13, 200)
ax.plot(x_line, slope * x_line + intercept, 'tomato', linewidth=1.8, alpha=0.85,
        label=f'Weighted ODR  slope={slope:.2f},  R={r:.2f}')

stats_text = (
    f'Bias:        {bias:+.2f} Gyr\n'
    f'Resid std:   {resid_std:.2f} Gyr\n'
    f'NMAD:        {nmad:.2f} Gyr\n'
    f'Wtd residual:{w_resid:+.2f} Gyr\n'
    f'Median σ_age: −{comp["age_err_lo"].median():.2f} / +{comp["age_err_hi"].median():.2f} Gyr'
)
ax.text(0.04, 0.97, stats_text,
        transform=ax.transAxes, fontsize=9, va='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.88))

ax.set_xlim(*lim)
ax.set_ylim(0, 10.5)
ax.set_xlabel('Catalog Age (Gyr)', fontsize=12)
ax.set_ylabel('MCMC Median Age (Gyr)', fontsize=12)
ax.set_title('Isochrone Fit Age vs Catalog Age — MCMC Posteriors', fontsize=13, pad=12)
ax.legend(fontsize=10, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(_OUTPUTS, 'age_comparison.png'), dpi=150)
plt.close()
print("Saved: age_comparison.png")


# ── 3. My Fit vs James's Fit ──────────────────────────────────────────────────
james = pd.read_csv(os.path.join(_ANALYSIS, 'James_interpolate_20260414_161013_candidate_fits.csv'))
james['age_gyr_james'] = james['age_yr'] / 1e9

vs_james = rel.merge(james[['hostname', 'age_gyr_james']], on='hostname', how='inner')
vs_james = vs_james[np.isfinite(vs_james['age_gyr_james'])].copy()

xj    = vs_james['age_gyr_james'].values
yj    = vs_james['age_plot'].values
yerrj = ((vs_james['age_err_lo'] + vs_james['age_err_hi']) / 2).values

_, _, rj, _, _ = stats.linregress(xj, yj)
resid_j  = yj - xj
bias_j   = resid_j.mean()
nmad_j   = 1.4826 * np.median(np.abs(resid_j - np.median(resid_j)))

# ODR weighted by MCMC uncertainties (James has no published errors so xerr=1.5 uniform)
xerr_j    = np.full(len(xj), 1.5)
odr_data_j  = odr.Data(xj, yj, wd=1/xerr_j**2, we=1/yerrj**2)
odr_run_j   = odr.ODR(odr_data_j, odr.Model(lambda B, x: B[0]*x + B[1]), beta0=[1.0, 0.0]).run()
slope_j, intercept_j = odr_run_j.beta

print(f"\nJames comparison ({len(vs_james)} stars):")
print(f"  Bias:        {bias_j:+.2f} Gyr")
print(f"  NMAD:        {nmad_j:.2f} Gyr")
print(f"  Pearson R:   {rj:.3f}")
print(f"  ODR slope / intercept: {slope_j:.3f} / {intercept_j:.3f}")

fig, ax = plt.subplots(figsize=(9, 8))

ax.errorbar(xj, yj,
            yerr=[vs_james['age_err_lo'].values, vs_james['age_err_hi'].values],
            fmt='none', ecolor='0.65', alpha=0.35, linewidth=0.6, zorder=3)

sc2 = ax.scatter(xj, yj,
                 c=vs_james['mass_msun'], cmap='plasma',
                 vmin=mass_min, vmax=mass_max,
                 s=28, linewidths=0.3, edgecolors='0.3', zorder=4,
                 label=f'Reliable fits  (n={len(vs_james)})')

cbar2 = plt.colorbar(sc2, ax=ax, pad=0.02, fraction=0.046)
cbar2.set_label('Stellar Mass (M☉)', fontsize=10)

lim = (0, 14)
ax.plot(lim, lim, 'k--', linewidth=1, alpha=0.5, label='1:1')
x_line_j = np.linspace(0, 13, 200)
ax.plot(x_line_j, slope_j * x_line_j + intercept_j, 'tomato', linewidth=1.8, alpha=0.85,
        label=f'Weighted ODR  slope={slope_j:.2f},  R={rj:.2f}')

stats_text_j = (f'Bias:  {bias_j:+.2f} Gyr\n'
                f'NMAD:  {nmad_j:.2f} Gyr\n'
                f'R:     {rj:.3f}')
ax.text(0.04, 0.97, stats_text_j,
        transform=ax.transAxes, fontsize=10, va='top', family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.88))

ax.set_xlim(*lim)
ax.set_ylim(0, 10.5)
ax.set_xlabel("James's Fit Age (Gyr)", fontsize=12)
ax.set_ylabel('My MCMC Median Age (Gyr)', fontsize=12)
ax.set_title("My Fit vs James's Fit — Exoplanet Host Stars", fontsize=13, pad=12)
ax.legend(fontsize=10, framealpha=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(_OUTPUTS, 'age_vs_james.png'), dpi=150)
plt.close()
print("Saved: age_vs_james.png")


# ── 4. Age Uncertainty Distribution ───────────────────────────────────────────
# Describes the spread: how well-constrained are the MCMC posteriors?
has_mcmc = rel[rel['accept_fr'].notna()].copy() if 'accept_fr' in rel.columns else rel.copy()
sym_err  = (has_mcmc['age_err_lo'] + has_mcmc['age_err_hi']) / 2   # avg ±σ per star

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: histogram of posterior widths
ax0 = axes[0]
ax0.hist(sym_err, bins=30, color='steelblue', edgecolor='white', linewidth=0.5)
ax0.axvline(sym_err.median(), color='tomato', linewidth=1.8, linestyle='--',
            label=f'Median = {sym_err.median():.2f} Gyr')
ax0.axvline(sym_err.mean(), color='orange', linewidth=1.4, linestyle=':',
            label=f'Mean   = {sym_err.mean():.2f} Gyr')
ax0.set_xlabel('Age Uncertainty ½(σ₋ + σ₊) (Gyr)', fontsize=11)
ax0.set_ylabel('Number of Stars', fontsize=11)
ax0.set_title('MCMC Age Uncertainty Distribution', fontsize=12)
ax0.legend(fontsize=9)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)

# Right: posterior width vs fitted age (does uncertainty grow with age?)
ax1 = axes[1]
sc3 = ax1.scatter(has_mcmc['age_plot'], sym_err,
                  c=has_mcmc['mass_msun'], cmap='plasma',
                  vmin=mass_min, vmax=mass_max,
                  s=18, alpha=0.6, linewidths=0.2, edgecolors='0.3')
cbar3 = plt.colorbar(sc3, ax=ax1, pad=0.02, fraction=0.046)
cbar3.set_label('Stellar Mass (M☉)', fontsize=9)

# Bin-median trend
age_bins  = np.arange(0, 11, 1)
bin_meds  = [sym_err[(has_mcmc['age_plot'] >= a) & (has_mcmc['age_plot'] < a+1)].median()
             for a in age_bins]
bin_cents = age_bins + 0.5
valid     = [i for i, v in enumerate(bin_meds) if not np.isnan(v)]
ax1.plot([bin_cents[i] for i in valid], [bin_meds[i] for i in valid],
         'k-o', linewidth=1.5, markersize=5, label='Bin median', zorder=5)

# Stats annotation
pct_wide = 100 * (sym_err > 2.0).mean()
ax1.text(0.04, 0.97,
         f'Std of fit ages:   {has_mcmc["age_plot"].std():.2f} Gyr\n'
         f'Median σ_age:      {sym_err.median():.2f} Gyr\n'
         f'Frac with σ > 2 Gyr: {pct_wide:.0f}%',
         transform=ax1.transAxes, fontsize=9, va='top', family='monospace',
         bbox=dict(boxstyle='round,pad=0.4', fc='white', alpha=0.88))

ax1.set_xlabel('MCMC Median Age (Gyr)', fontsize=11)
ax1.set_ylabel('Age Uncertainty ½(σ₋ + σ₊) (Gyr)', fontsize=11)
ax1.set_title('Age Uncertainty vs Fitted Age', fontsize=12)
ax1.legend(fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(_OUTPUTS, 'age_uncertainty.png'), dpi=150)
plt.close()
print("Saved: age_uncertainty.png")
