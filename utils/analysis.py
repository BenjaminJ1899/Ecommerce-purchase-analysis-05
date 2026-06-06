import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# 确保从项目根目录或 utils/ 目录运行都能正确导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from data_loader import load_cleaned_data, load_raw_data

OUT_DIR = r'e:\SQL-xuexi\客户购买行为数据集-电子商务'

# ============================================================
# 1. DATA CLEANING
# ============================================================
print("=" * 60)
print("1. DATA CLEANING")
print("=" * 60)

raw = load_raw_data()
print(f'Raw data: {raw.shape}, Purchase rate: {raw["PurchaseStatus"].mean()*100:.1f}%')

# Report what will be cleaned
neg_cols = ['NumberOfPurchases', 'TimeSpentOnWebsite', 'CustomerTenureYears', 'LastPurchaseDaysAgo']
for col in neg_cols:
    neg = (raw[col] < 0).sum()
    if neg > 0:
        print(f'  {col}: {neg} negative values will be fixed -> 0')
age_bad = (raw['Age'] < 17).sum()
print(f'  Age<17: {age_bad} rows will be removed')

df = load_cleaned_data()
print(f'Clean data: {df.shape}')

# ============================================================
# 2. CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("2. FEATURE CORRELATION WITH PURCHASE")
print("=" * 60)

num_cols = ['Age','AnnualIncome','NumberOfPurchases','TimeSpentOnWebsite',
            'CustomerTenureYears','LastPurchaseDaysAgo','DiscountsAvailed',
            'SessionCount','CustomerSatisfaction','PurchaseStatus']
corr = df[num_cols].corr()['PurchaseStatus'].drop('PurchaseStatus').sort_values(key=abs, ascending=False)
for col, val in corr.items():
    direction = '+' if val > 0 else '-'
    print(f'  {col:>25}: {val:+.4f} ({direction})')

# ============================================================
# 3. PURCHASE BY CATEGORY
# ============================================================
print("\n" + "=" * 60)
print("3. PURCHASE RATE BY SEGMENTS")
print("=" * 60)

for col in ['ProductCategory','ReferralSource','CustomerSegment','Gender',
            'PreferredDevice','Region','LoyaltyProgram']:
    print(f'\n--- {col} ---')
    grp = df.groupby(col)['PurchaseStatus'].agg(['mean','count'])
    grp['mean'] = grp['mean'] * 100
    grp = grp.sort_values('mean', ascending=False)
    for idx, row in grp.iterrows():
        print(f'  {str(idx):>15}: {row["mean"]:5.1f}% (n={row["count"]:>,})')

# ============================================================
# 4. VIP vs Regular deep dive
# ============================================================
print("\n" + "=" * 60)
print("4. CUSTOMER SEGMENT DEEP DIVE")
print("=" * 60)

for seg in ['VIP','Premium','Regular']:
    sdf = df[df['CustomerSegment'] == seg]
    print(f'\n  {seg} (n={len(sdf):,}, {len(sdf)/len(df)*100:.1f}%):')
    print(f'    Purchase rate: {sdf["PurchaseStatus"].mean()*100:.1f}%')
    print(f'    Avg purchases: {sdf["NumberOfPurchases"].mean():.1f}')
    print(f'    Avg income: ${sdf["AnnualIncome"].mean():.0f}')
    print(f'    Avg satisfaction: {sdf["CustomerSatisfaction"].mean():.1f}')
    print(f'    Avg time on site: {sdf["TimeSpentOnWebsite"].mean():.1f}')
    print(f'    Loyalty program: {sdf["LoyaltyProgram"].mean()*100:.1f}%')

# ============================================================
# 5. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("5. GENERATING CHARTS")
print("=" * 60)

# 5a. Purchase by category + referral
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cat_rates = df.groupby('ProductCategory')['PurchaseStatus'].mean().sort_values() * 100
axes[0].barh(range(len(cat_rates)), cat_rates.values, color=['#4C72B0','#55A868','#C44E52','#DD8452','#937860'])
axes[0].set_yticks(range(len(cat_rates)))
axes[0].set_yticklabels(cat_rates.index)
axes[0].set_title('Purchase Rate by Category', fontsize=12, fontweight='bold')
for i, v in enumerate(cat_rates.values):
    axes[0].text(v+0.5, i, f'{v:.1f}%', va='center', fontsize=10)

ref_rates = df.groupby('ReferralSource')['PurchaseStatus'].mean().sort_values() * 100
axes[1].barh(range(len(ref_rates)), ref_rates.values, color=['#4C72B0','#55A868','#C44E52','#DD8452','#937860'])
axes[1].set_yticks(range(len(ref_rates)))
axes[1].set_yticklabels(ref_rates.index)
axes[1].set_title('Purchase Rate by Referral Source', fontsize=12, fontweight='bold')
for i, v in enumerate(ref_rates.values):
    axes[1].text(v+0.5, i, f'{v:.1f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_purchase_by_segment.png'), dpi=150)
plt.close()
print('  -> fig_purchase_by_segment.png')

# 5b. Feature importance (correlation bar chart)
fig, ax = plt.subplots(figsize=(10, 6))
colors_bar = ['#C44E52' if v < 0 else '#4C72B0' for v in corr.values]
bars = ax.barh([c[:30] for c in corr.index], corr.values, color=colors_bar)
ax.set_title('Feature Correlation with Purchase', fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
for bar, val in zip(bars, corr.values):
    ax.text(bar.get_width() + 0.002 if val > 0 else bar.get_width() - 0.008,
            bar.get_y() + bar.get_height()/2, f'{val:+.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_correlation.png'), dpi=150)
plt.close()
print('  -> fig_correlation.png')

# 5c. Income vs Purchases scatter by segment
fig, ax = plt.subplots(figsize=(10, 7))
seg_colors = {'VIP': '#C44E52', 'Premium': '#4C72B0', 'Regular': '#55A868'}
sample = df.sample(5000, random_state=42)
for seg in ['VIP','Premium','Regular']:
    sdf = sample[sample['CustomerSegment'] == seg]
    ax.scatter(sdf['AnnualIncome'], sdf['NumberOfPurchases'],
              c=seg_colors[seg], label=seg, alpha=0.5, s=10)
ax.set_xlabel('Annual Income ($)', fontsize=12)
ax.set_ylabel('Number of Purchases', fontsize=12)
ax.set_title('Income vs Purchases by Customer Segment', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_income_vs_purchases.png'), dpi=150)
plt.close()
print('  -> fig_income_vs_purchases.png')

# 5d. Satisfaction distribution by purchase status
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sat_dist = df.groupby(['CustomerSatisfaction','PurchaseStatus']).size().unstack()
sat_dist_pct = sat_dist.div(sat_dist.sum(axis=1), axis=0) * 100
x = np.arange(1, 6)
w = 0.35
axes[0].bar(x - w/2, sat_dist[1], w, label='Purchased', color='#4C72B0')
axes[0].bar(x + w/2, sat_dist[0], w, label='Not Purchased', color='#C44E52')
axes[0].set_xticks(x)
axes[0].set_xlabel('Satisfaction Score', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Satisfaction vs Purchase', fontsize=12, fontweight='bold')
axes[0].legend()

for dev in ['Mobile','Desktop','Tablet']:
    ddf = df[df['PreferredDevice'] == dev]
    print(f'  {dev}: purchase_rate={ddf["PurchaseStatus"].mean()*100:.1f}%  avg_time={ddf["TimeSpentOnWebsite"].mean():.1f}')

dev_rates = df.groupby('PreferredDevice')['PurchaseStatus'].mean().sort_values() * 100
axes[1].barh(range(len(dev_rates)), dev_rates.values, color=['#4C72B0','#55A868','#DD8452'])
axes[1].set_yticks(range(len(dev_rates)))
axes[1].set_yticklabels(dev_rates.index)
axes[1].set_title('Purchase Rate by Device', fontsize=12, fontweight='bold')
for i, v in enumerate(dev_rates.values):
    axes[1].text(v+0.5, i, f'{v:.1f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig_satisfaction_device.png'), dpi=150)
plt.close()
print('  -> fig_satisfaction_device.png')

# ============================================================
# 6. KEY INSIGHTS
# ============================================================
print("\n" + "=" * 60)
print("6. KEY INSIGHTS")
print("=" * 60)

best_cat = df.groupby('ProductCategory')['PurchaseStatus'].mean().idxmax()
best_cat_rate = df.groupby('ProductCategory')['PurchaseStatus'].mean().max() * 100
best_ref = df.groupby('ReferralSource')['PurchaseStatus'].mean().idxmax()
best_ref_rate = df.groupby('ReferralSource')['PurchaseStatus'].mean().max() * 100
vip_rate = df[df['CustomerSegment']=='VIP']['PurchaseStatus'].mean()*100
reg_rate = df[df['CustomerSegment']=='Regular']['PurchaseStatus'].mean()*100
loyal_rate = df[df['LoyaltyProgram']==1]['PurchaseStatus'].mean()*100
non_loyal_rate = df[df['LoyaltyProgram']==0]['PurchaseStatus'].mean()*100
desktop_rate = df[df['PreferredDevice']=='Desktop']['PurchaseStatus'].mean()*100
mobile_rate = df[df['PreferredDevice']=='Mobile']['PurchaseStatus'].mean()*100

print(f"""
  DATA: {len(df):,} customers, purchase rate {df['PurchaseStatus'].mean()*100:.1f}%

  TOP PURCHASE DRIVERS (by correlation):
    1. LastPurchaseDaysAgo ({corr['LastPurchaseDaysAgo']:+.3f}) - recency dominates everything
    2. CustomerSatisfaction ({corr['CustomerSatisfaction']:+.3f}) - happier = more likely
    3. AnnualIncome ({corr['AnnualIncome']:+.3f}) - higher income, higher purchase rate
    4. NumberOfPurchases ({corr['NumberOfPurchases']:+.3f}) - past behavior predicts future

  SEGMENT DIFFERENCES:
    VIP: {vip_rate:.1f}% purchase rate
    Regular: {reg_rate:.1f}% purchase rate
    Gap: {vip_rate-reg_rate:.1f}pp

  LOYALTY PROGRAM:
    Loyalty members: {loyal_rate:.1f}% purchase rate
    Non-members: {non_loyal_rate:.1f}% purchase rate
    Lift: {loyal_rate-non_loyal_rate:.1f}pp

  BEST CATEGORY: {best_cat} ({best_cat_rate:.1f}%)
  BEST CHANNEL: {best_ref} ({best_ref_rate:.1f}%)
  DESKTOP vs MOBILE: {desktop_rate:.1f}% vs {mobile_rate:.1f}%
""")

print("DONE! All charts saved.")
