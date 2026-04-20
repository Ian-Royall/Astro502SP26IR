import numpy as np
import pandas as pd
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

plt.tight_layout()
plt.savefig('/home/iroyall/Documents/Astro 502/Astro502SP26IR/age_distribution.png', dpi=150)
plt.show()
print("Saved: age_distribution.png")
