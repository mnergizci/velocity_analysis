#!/usr/bin/env python3
from lics_unwrap import *
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from modules import *
from sklearn.linear_model import LinearRegression

##fault
faults = gpd.read_file('/home/users/mnergiz/1.gmt_workout/1.turkey_paper/data/GEM_TR.shp')
thresh = 0.99 # mm/yr
###3D
vn_3d=load_tif2xr('decomposed/north_3D.tif')
ve_3d=load_tif2xr('decomposed/east_3D.tif')
vu_3d=load_tif2xr('decomposed/up_3D.tif')
vn_sig_3d=load_tif2xr('decomposed/north_sigma_3D.tif')
ve_sig_3d=load_tif2xr('decomposed/east_sigma_3D.tif')
vu_sig_3d=load_tif2xr('decomposed/up_sigma_3D.tif')

###2D
ve_2d=load_tif2xr('decomposed/east_2D.tif')
vu_2d=load_tif2xr('decomposed/up_2D.tif')
ve_sig_2d=load_tif2xr('decomposed/east_sigma_2D.tif')
vu_sig_2d=load_tif2xr('decomposed/up_sigma_2D.tif')

###Combined
ve_comp=load_tif2xr('decomposed/east_combined.tif')
vu_comp=load_tif2xr('decomposed/up_combined.tif')
ve_sig_comp=load_tif2xr('decomposed/east_sigma_combined.tif')
vu_sig_comp=load_tif2xr('decomposed/up_sigma_combined.tif')

##gnss 
vn_gnss_krigin=load_tif2xr('gps_kriging/external_drift/vn_interpolated_upsampled.tif')
#gnss-pandas
GNSS_AHB_2D_nnr=pd.read_csv('GNSS_processing/AHB_GPS/ahb_v5pt3pt1_2D_31-Mar-2025_eu.dat', delim_whitespace=True)
GNSS_AHB_3D_nnr=pd.read_csv('GNSS_processing/AHB_GPS/ahb_v5pt3pt1_3D_31-Mar-2025_eu.dat', delim_whitespace=True)
region_TR = [24.5, 45.5, 32, 43]
lon_min, lon_max, lat_min, lat_max = region_TR

# Filter both DataFrames
df_TR_3D = GNSS_AHB_3D_nnr[
    (GNSS_AHB_3D_nnr['lon'] >= lon_min) &
    (GNSS_AHB_3D_nnr['lon'] <= lon_max) &
    (GNSS_AHB_3D_nnr['lat'] >= lat_min) &
    (GNSS_AHB_3D_nnr['lat'] <= lat_max)
]

df_TR_2D = GNSS_AHB_2D_nnr[
    (GNSS_AHB_2D_nnr['lon'] >= lon_min) &
    (GNSS_AHB_2D_nnr['lon'] <= lon_max) &
    (GNSS_AHB_2D_nnr['lat'] >= lat_min) &
    (GNSS_AHB_2D_nnr['lat'] <= lat_max)
]

# Append 2D to 3D
df_TR_combined = df_TR_3D.append(df_TR_2D, ignore_index=True)

# Restrict the coordinates to 33–42 for both lat and lon
df_TR_combined = df_TR_combined[
    (df_TR_combined['lat'] >= 33.5) & (df_TR_combined['lat'] <= 43) &
    (df_TR_combined['lon'] >= 33.5) & (df_TR_combined['lon'] <= 43)
]

# Flatten and clean NaNs for each component
vn_sig_values = vn_sig_3d.values.flatten()
vn_sig_values = vn_sig_values[np.isfinite(vn_sig_values)]

ve_sig_values = ve_sig_3d.values.flatten()
ve_sig_values = ve_sig_values[np.isfinite(ve_sig_values)]

vu_sig_values = vu_sig_3d.values.flatten()
vu_sig_values = vu_sig_3d.values.flatten()
vu_sig_values = vu_sig_values[np.isfinite(vu_sig_values)]

# Titles and data
components = ['Ve', 'Vn', 'Vu']
values = [ve_sig_values, vn_sig_values, vu_sig_values]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

for ax, val, title in zip(axes, values, components):
    mean_val = np.mean(val)
    std_val = np.std(val)
    pct_high = 100 * np.sum(val > 0.99) / len(val)

    ax.hist(val, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val:.2f}')
    # ax.axvline(mean_val + 2*std_val, color='orange', linestyle='--', linewidth=1, label=f'+1σ = {mean_val + 2*std_val:.2f}')
    # ax.axvline(mean_val - 2*std_val, color='orange', linestyle='--', linewidth=1, label=f'-1σ = {mean_val - 2*std_val:.2f}')
    ax.axvline(0.99, color='purple', linestyle=':', linewidth=1.5, label='Thres = 0.99')


    ax.set_title(f'σ {title}', fontsize=12)
    # ax.set_xlabel('Uncertainty (mm/yr)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=7, loc='lower right')

    # Add percentage annotation
    ax.text(0.95, 0.95, f'>0.99 mm/yr: {pct_high:.1f}%', 
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# axes[0].set_ylabel('Number of pixels', fontsize=10)

plt.tight_layout()
plt.savefig('v_sigma_values.png', dpi=300)
plt.show()

# Flatten and clean NaNs for each component
vn_sig_values = vn_sig_3d.values.flatten()
vn_sig_values = vn_sig_values[np.isfinite(vn_sig_values)]

ve_sig_comp_values = ve_sig_comp.values.flatten()
ve_sig_comp_values = ve_sig_comp_values[np.isfinite(ve_sig_comp_values)]

vu_sig_comp_values = vu_sig_comp.values.flatten()
vu_sig_comp_values = vu_sig_comp_values[np.isfinite(vu_sig_comp_values)]

# Titles and data
components = ['Ve', 'Vn', 'Vu']
values = [ve_sig_comp_values, vn_sig_values, vu_sig_comp_values]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)

for ax, val, title in zip(axes, values, components):
    mean_val = np.mean(val)
    std_val = np.std(val)
    pct_high = 100 * np.sum(val > 0.99) / len(val)

    ax.hist(val, bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val:.2f}')
    # ax.axvline(mean_val + 2*std_val, color='orange', linestyle='--', linewidth=1, label=f'+1σ = {mean_val + 2*std_val:.2f}')
    # ax.axvline(mean_val - 2*std_val, color='orange', linestyle='--', linewidth=1, label=f'-1σ = {mean_val - 2*std_val:.2f}')
    ax.axvline(0.99, color='purple', linestyle=':', linewidth=1.5, label='Thres = 0.99')


    ax.set_title(f'σ {title}', fontsize=12)
    # ax.set_xlabel('Uncertainty (mm/yr)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=7, loc='lower right')

    # Add percentage annotation
    ax.text(0.95, 0.95, f'>0.99 mm/yr: {pct_high:.1f}%', 
            transform=ax.transAxes, ha='right', va='top',
            fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

# axes[0].set_ylabel('Number of pixels', fontsize=10)

plt.tight_layout()
plt.savefig('v__combined_sigma_values.png', dpi=300)
plt.show()

# ── generate two masked arrays ──────────────────────────────
thresh = 0.99 # mm/yr
low_unc  = vn_sig_3d.where(vn_sig_3d <  thresh)
high_unc = vn_sig_3d.where(vn_sig_3d >= thresh)

# ── stats for annotation ────────────────────────────────────
total   = np.isfinite(vn_sig_3d.values).sum()
n_low   = np.isfinite(low_unc.values).sum()
n_high  = np.isfinite(high_unc.values).sum()

pct_low  = 100 * n_low  / total
pct_high = 100 * n_high / total

# ── plotting ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharex=True, sharey=True)

for ax, da, title, pct in zip(
        axes,
        [low_unc, high_unc],
        [f'σ < {thresh} mm/yr', f'σ ≥ {thresh} mm/yr'],
        [pct_low, pct_high]):

    im = da.plot(ax=ax, cmap='viridis', vmin=0, vmax=1, add_colorbar=False)

    # overlay GNSS uncertainties (optional)
    ax.scatter(df_TR_combined['lon'], df_TR_combined['lat'],
               c=df_TR_combined['snorth'], cmap='viridis',
               vmin=0, vmax=1, edgecolor='black', s=20)

    ax.set_xlim(33, 43)
    ax.set_title(title, fontsize=14)
    # ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    faults.plot(ax=ax, edgecolor='grey', linewidth=0.2)
    
    # percentage text
    ax.text(0.02, 0.96, f'{pct:.1f} %', transform=ax.transAxes,
            ha='left', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.set_aspect("equal", adjustable="box")   # optional, keeps squares

# common colour-bar
cbar_ax = fig.add_axes([0.263, 0.07, 0.5, 0.025])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, orientation='horizontal').set_label('Uncertainty (mm/yr)')

plt.tight_layout(rect=[0,0.12,1,1])  # leave room for colour-bar
# plt.show()
plt.savefig('upperandlower_Vn_sig0.99_thres.png', dpi=300)


# ── generate two masked arrays ──────────────────────────────
thresh = 0.99 # mm/yr
low_unc  = ve_sig_comp.where(ve_sig_comp <  thresh)
high_unc = ve_sig_comp.where(ve_sig_comp >= thresh)

# ── stats for annotation ────────────────────────────────────
total   = np.isfinite(ve_sig_comp.values).sum()
n_low   = np.isfinite(low_unc.values).sum()
n_high  = np.isfinite(high_unc.values).sum()

pct_low  = 100 * n_low  / total
pct_high = 100 * n_high / total

# ── plotting ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharex=True, sharey=True)

for ax, da, title, pct in zip(
        axes,
        [low_unc, high_unc],
        [f'σ < {thresh} mm/yr', f'σ ≥ {thresh} mm/yr'],
        [pct_low, pct_high]):

    im = da.plot(ax=ax, cmap='viridis', vmin=0, vmax=1, add_colorbar=False)

    # overlay GNSS uncertainties (optional)
    ax.scatter(df_TR_combined['lon'], df_TR_combined['lat'],
               c=df_TR_combined['seast'], cmap='viridis',
               vmin=0, vmax=1, edgecolor='black', s=20)

    ax.set_xlim(33, 43)
    ax.set_title(title, fontsize=14)
    # ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    faults.plot(ax=ax, edgecolor='grey', linewidth=0.2)
    
    # percentage text
    ax.text(0.02, 0.96, f'{pct:.1f} %', transform=ax.transAxes,
            ha='left', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.set_aspect("equal", adjustable="box")   # optional, keeps squares

# common colour-bar
cbar_ax = fig.add_axes([0.263, 0.07, 0.5, 0.025])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, orientation='horizontal').set_label('Uncertainty (mm/yr)')

plt.tight_layout(rect=[0,0.12,1,1])  # leave room for colour-bar
# plt.show()
plt.savefig('upperandlower_Ve_sig0.99_thres.png', dpi=300)


##
# ── generate two masked arrays ──────────────────────────────
thresh = 0.99 # mm/yr
low_unc  = vu_sig_comp.where(vu_sig_comp <  thresh)
high_unc = vu_sig_comp.where(vu_sig_comp >= thresh)

# ── stats for annotation ────────────────────────────────────
total   = np.isfinite(vu_sig_comp.values).sum()
n_low   = np.isfinite(low_unc.values).sum()
n_high  = np.isfinite(high_unc.values).sum()

pct_low  = 100 * n_low  / total
pct_high = 100 * n_high / total

# ── plotting ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharex=True, sharey=True)

for ax, da, title, pct in zip(
        axes,
        [low_unc, high_unc],
        [f'σ < {thresh} mm/yr', f'σ ≥ {thresh} mm/yr'],
        [pct_low, pct_high]):

    im = da.plot(ax=ax, cmap='viridis', vmin=0, vmax=1, add_colorbar=False)

    # overlay GNSS uncertainties (optional)
    ax.scatter(df_TR_combined['lon'], df_TR_combined['lat'],
               c=df_TR_combined['sup'], cmap='viridis',
               vmin=0, vmax=1, edgecolor='black', s=20)

    ax.set_xlim(33, 43)
    ax.set_title(title, fontsize=14)
    # ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    faults.plot(ax=ax, edgecolor='grey', linewidth=0.2)
    
    # percentage text
    ax.text(0.02, 0.96, f'{pct:.1f} %', transform=ax.transAxes,
            ha='left', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    ax.set_aspect("equal", adjustable="box")   # optional, keeps squares

# common colour-bar
cbar_ax = fig.add_axes([0.263, 0.07, 0.5, 0.025])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, orientation='horizontal').set_label('Uncertainty (mm/yr)')

plt.tight_layout(rect=[0,0.12,1,1])  # leave room for colour-bar
# plt.show()
plt.savefig('upperandlower_Vu_sig0.99_thres.png', dpi=300)


###GNSS-Comparison
###open the files
vn_3d_tif = OpenTif('decomposed/north_3D.tif')
ve_3d_tif = OpenTif('decomposed/east_combined.tif')
vu_3d_tif = OpenTif('decomposed/up_combined.tif')

vn_sig_3d_tif = OpenTif('decomposed/north_sigma_3D.tif')
ve_sig_3d_tif = OpenTif('decomposed/east_sigma_combined.tif')
vu_sig_3d_tif = OpenTif('decomposed/up_sigma_combined.tif')

half_window = 100
motion_ns = [vn_3d_tif.extract_pixel_value2(x, y, half_window)[0] for x, y in zip(df_TR_combined['lon'], df_TR_combined['lat'])]
motion_ew = [ve_3d_tif.extract_pixel_value2(x, y, half_window)[0] for x, y in zip(df_TR_combined['lon'], df_TR_combined['lat'])]
motion_ud = [vu_3d_tif.extract_pixel_value2(x, y, half_window)[0] for x, y in zip(df_TR_combined['lon'], df_TR_combined['lat'])]

half_window = 100
motion_ns_sig = [vn_sig_3d_tif.extract_pixel_value2(x, y, half_window)[0] for x, y in zip(df_TR_combined['lon'], df_TR_combined['lat'])]
motion_ew_sig = [ve_sig_3d_tif.extract_pixel_value2(x, y, half_window)[0] for x, y in zip(df_TR_combined['lon'], df_TR_combined['lat'])]
motion_ud_sig = [vu_sig_3d_tif.extract_pixel_value2(x, y, half_window)[0] for x, y in zip(df_TR_combined['lon'], df_TR_combined['lat'])]


df_TR_combined['vn_3d']=motion_ns
df_TR_combined['ve_3d']=motion_ew
df_TR_combined['vu_3d']=motion_ud

df_TR_combined['vn_sig_3d']=motion_ns_sig
df_TR_combined['ve_sig_3d']=motion_ew_sig
df_TR_combined['vu_sig_3d']=motion_ud_sig

df_BOI_GNSS_vn_comp=df_TR_combined[['lat','lon','vnorth_eu','snorth','vn_3d', 'vn_sig_3d']]
df_BOI_GNSS_vn_comp = df_BOI_GNSS_vn_comp.dropna(subset=['vnorth_eu', 'vn_3d', 'vn_sig_3d'])
df_BOI_GNSS_vn_comp['Vn-diff'] = (df_BOI_GNSS_vn_comp['vnorth_eu'] - df_BOI_GNSS_vn_comp['vn_3d'])

df_BOI_GNSS_ve_comp=df_TR_combined[['lat','lon','veast_eu','seast','ve_3d', 've_sig_3d']]
df_BOI_GNSS_ve_comp = df_BOI_GNSS_ve_comp.dropna(subset=['veast_eu', 've_3d', 've_sig_3d'])
df_BOI_GNSS_ve_comp['Ve-diff'] = (df_BOI_GNSS_ve_comp['veast_eu'] - df_BOI_GNSS_ve_comp['ve_3d'])

df_BOI_GNSS_vu_comp=df_TR_combined[['lat','lon','vup','sup','vu_3d', 'vu_sig_3d']]
df_BOI_GNSS_vu_comp = df_BOI_GNSS_vu_comp.dropna(subset=['vup', 'vu_3d', 'vu_sig_3d'])
df_BOI_GNSS_vu_comp['Vu-diff'] = (df_BOI_GNSS_vu_comp['vup'] - df_BOI_GNSS_vu_comp['vu_3d'])
