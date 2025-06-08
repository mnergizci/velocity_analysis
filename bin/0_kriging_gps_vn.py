#!/usr/bin/env python3

#####################
# Applying the PyKrige package to interpolate point data into surface data.
# Qi Ou, University of Leeds
# 3 Sep 2022
##############

import geopandas as gpd
import shapely.speedups
# gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
# shapely.speedups.enable()
from pykrige.uk import UniversalKriging
from pykrige.kriging_tools import write_asc_grid
from matplotlib import cm
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import pandas as pd
import argparse
import os
import subprocess

def load_and_prepare_gps(ahb_2d_path, ahb_3d_path, lon_min, lon_max, lat_min, lat_max, sn_threshold=1.0):
    """
    Load GNSS 2D and 3D data from compiled GNSS C.Rollins, filter by bounding box, clean and return a GeoDataFrame.
    
    Parameters:
    - ahb_2d_path (str): Path to 2D GNSS dataset
    - ahb_3d_path (str): Path to 3D GNSS dataset
    - lon_min, lon_max, lat_min, lat_max (float): Bounding box for region of interest
    
    Returns:
    - gps (GeoDataFrame): Cleaned and filtered GPS data ready for kriging
    """
    
    # Load datasets
    GNSS_AHB_2D_nnr = pd.read_csv(ahb_2d_path, delim_whitespace=True)
    GNSS_AHB_3D_nnr = pd.read_csv(ahb_3d_path, delim_whitespace=True)
    
    # Filter by bounding box
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
    
    # Combine both
    df = pd.concat([df_TR_3D, df_TR_2D], ignore_index=True)
    
    # Drop rows where sn > threshold
    df = df[df['snorth'] <= sn_threshold]
    
    # Prepare required columns
    df['station'] = df['names'] if 'names' in df.columns else ['STN_{:03d}'.format(i) for i in range(len(df))]
    
    df['ve'] = df['veast_eu']
    df['vn'] = df['vnorth_eu']
    df['vu'] = df.get('vup', 0.0)
    
    df['se'] = df['seast']
    df['sn'] = df['snorth']
    df['su'] = df.get('sup', 0.0)
    
    df['cen'] = 0.0
    df['geometry'] = df.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    
    gps = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    gps = gps[['station', 've', 'vn', 'vu', 'se', 'sn', 'su', 'cen', 'geometry']]
    
    return gps

def polyfit2d(x, y, z, order=3):
    """
    Fit a (default 3rd order) polynomial model predictions to GPS points
    @param x: lon of GPS points
    @param y: lat of GPS points
    @param z: velocity of GPS points
    @return: m: polynomial model
    """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z, rcond=None)
    return m


def polyval2d(x, y, m):
    """
    Evaluate polynomial model predictions at given locations based on the polynomial model
    @param x: x array
    @param y: y array
    @param m: polynomial model
    @return:  z array
    """
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z


def grid_gps_with_external_drift(gps_df, gridx, gridy, comp, zz):
    """ (as far as I can follow from the PyKrige documentation)
    Universal kriging with external polynomial drift
    @param gps_df: gps dataframe
    @param gridx: x locations for interpolation
    @param gridy: y locations for interpolation
    @param comp: which component (column name) in the GPS dataframe you want to interpolate
    @param zz: best-fit polynomial surface of the same dimension as the interpolated map
    @param m: polynomial model obtained from polyfit2d
    @return: interpolated velocity field and uncertainty field
    """
    # Kriging
    model = "spherical"
    # model = "linear"
    # model = "gaussian"
    UK = UniversalKriging(
        gps_df.geometry.x,
        gps_df.geometry.y,
        gps_df[comp],
        # drift_terms=["external_Z"],
        # external_drift=zz,
        # external_drift_x=gridx,
        # external_drift_y=gridy,
        variogram_model=model,
        weight=True,
        nlags=20,
        enable_plotting=False,
        # exact_values=False
        # coordinates_type="geographic"  # not yet implemented for universal kriging
    )

    vel_interpolated, var_interpolated = UK.execute("grid", gridx, gridy)

    return vel_interpolated, np.sqrt(var_interpolated)


def grid_gps_from_detrended_gps_points(gps_df, gridx, gridy, comp, zz, m):
    """ (manual implementation of universal kriging based on my understanding of the theory)
    1. Fit polynomial trend to GNSS data
    2. Remove the polynomial predictions at GNSS locations from the GNSS data
    3. Interpolate the GNSS deviations from the polynomial removed
    4. Add back the removed polynomial surface to the interpolated map
    @param gps_df: gps dataframe
    @param gridx: x locations for interpolation
    @param gridy: y locations for interpolation
    @param comp: which component (column name) in the GPS dataframe you want to interpolate
    @param zz: best-fit polynomial surface of the same dimension as the interpolated map
    @param m: polynomial model obtained from polyfit2d
    @return: interpolated velocity field and uncertainty field
    """
    # Detrend GPS points
    z = polyval2d(gps_df.geometry.x, gps_df.geometry.y, m)
    detrended_vel = gps_df[comp] - z  # z is evaluated at point locations, different from zz which is a grid

    # Kriging
    model = "spherical"
    # model = "gaussian"
    UK = UniversalKriging(
        gps_df.geometry.x,
        gps_df.geometry.y,
        detrended_vel,
        variogram_model=model,
        weight=True,
        nlags=20,
        enable_plotting=True,
        exact_values=False
        # coordinates_type="geographic"  # not yet implemented for universal kriging
    )

    vel_detrend_interpolated, var_detrend_interpolated = UK.execute("grid", gridx, gridy)

    return vel_detrend_interpolated + zz, np.sqrt(var_detrend_interpolated)


def grid_gps_with_specified_drift(gps_df, gridx, gridy, comp, m):
    """ (manual implementation of universal kriging based on my understanding of the theory)
    1. Fit polynomial trend to GNSS data
    2. Remove the polynomial predictions at GNSS locations from the GNSS data
    3. Interpolate the GNSS deviations from the polynomial removed
    4. Add back the removed polynomial surface to the interpolated map
    @param gps_df: gps dataframe
    @param gridx: x locations for interpolation
    @param gridy: y locations for interpolation
    @param comp: which component (column name) in the GPS dataframe you want to interpolate
    @param zz: best-fit polynomial surface of the same dimension as the interpolated map
    @param m: polynomial model obtained from polyfit2d
    @return: interpolated velocity field and uncertainty field
    """
    # Detrend GPS points
    z = polyval2d(gps_df.geometry.x, gps_df.geometry.y, m)
    # detrended_vel = gps_df[comp] - z  # z is evaluated at point locations, different from zz which is a grid

    # Kriging
    model = "spherical"
    # model = "gaussian"
    UK = UniversalKriging(
        gps_df.geometry.x,
        gps_df.geometry.y,
        gps_df[comp],
        drift_terms=["specified"],
        specified_drift=[z],
        variogram_model=model,
        weight=True,
        nlags=20,
        enable_plotting=True,
        exact_values=False
        # coordinates_type="geographic"  # not yet implemented for universal kriging
    )

    vel_interpolated, var_interpolated = UK.execute("grid", gridx, gridy)

    return vel_interpolated, np.sqrt(var_interpolated)

def grid_gps(gps_df, gridx, gridy, comp, out_dir, method="external_drift"):
    """
    Kriging process
    @param gps_df: GPS dataframe
    @param gridx: x locations where you want to derive an interpolated value
    @param gridy: y locations where you want to derive an interpolated value
    @param comp: which component (column name) in the GPS dataframe you want to interpolate
    @return: interpolated velocity field and uncertainty field
    """
    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(gps_df.geometry.x, gps_df.geometry.y, gps_df[comp])
    xx, yy = np.meshgrid(gridx, gridy)
    zz = polyval2d(xx, yy, m)

    # Plot polynomial fit to data points for universal kriging
    fit, ax = plt.subplots()
    im = ax.imshow(zz, extent=(west, east, south, north), origin='lower', vmin=vel_vmin, vmax=vel_vmax, cmap='RdBu_r')
    gps_df.plot(comp, ax=ax, vmin=vel_vmin, vmax=vel_vmax, edgecolor="w", cmap='RdBu_r')
    plt.colorbar(im, ax=ax)
    ax.set_xlim((west, east))
    ax.set_ylim((south, north))
    # ascending.plot(ax=ax, facecolor='None', edgecolor='w', alpha=1)  #facecolor='red',
    # descending.plot(ax=ax, facecolor='None', edgecolor='w', alpha=1)  #facecolor='blue',
    ax.set_title("Polynomial Surface Fit as Drift")
    plt.savefig(f'{out_dir}/polynomial_surface_fit.png')
    plt.close()
    
    if method == "external_drift":
        vel_interpolated, sig_interpolated = grid_gps_with_external_drift(gps_df, gridx, gridy, comp, zz)
    elif method == "detrended":
        vel_interpolated, sig_interpolated = grid_gps_from_detrended_gps_points(gps_df, gridx, gridy, comp, zz, m)
    elif method == "specified_drift":
        vel_interpolated, sig_interpolated = grid_gps_with_specified_drift(gps_df, gridx, gridy, comp, m)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return vel_interpolated, sig_interpolated


def plot_interpolation(vel_interpolated, sig_interpolated, out_dir):
    """ Plot the interpolated velocity and uncertainty maps """
    # plot vel_interpolated
    fig, ax = plt.subplots(2, figsize=(4.4, 6), sharex='all')
    im = ax[0].imshow(vel_interpolated, extent=(west, east, south, north), origin='lower', vmin=vel_vmin, vmax=vel_vmax, cmap='RdBu_r')  # , cmap=cm.bwr
    # gps.plot(ax=ax[0], facecolor="None", edgecolor='w') # to see GNSS location on interpolated field
    # gps[gps['sn']>0.7].plot(facecolor='None', edgecolor='r', ax=ax[0,0]) # to highlight GNSS with large uncertainties and see if they are causing local artefacts
    plt.colorbar(im, ax=ax[0], label="mm/yr")
    ax[0].set_title("V{}".format(component))
    ax[0].set_xlim((west, east))
    ax[0].set_ylim((south, north))
    # ascending.plot(ax=ax[0], facecolor='None', edgecolor='red', alpha=1)
    # descending.plot(ax=ax[0], facecolor='None', edgecolor='blue', alpha=1)

    # plot sig_interpolated
    sig_vmin = np.nanpercentile(sig_interpolated, 0.5)
    sig_vmax = np.nanpercentile(sig_interpolated, 99.5)
    im = ax[1].imshow(sig_interpolated, extent=(west, east, south, north), origin='lower', vmin=sig_vmin, vmax=sig_vmax)
    # gps.plot("sn", ax=ax[0, 1], vmin=vmin, vmax=vmax, edgecolor="w") # to see where the GNSS are
    plt.colorbar(im, ax=ax[1], label="mm/yr")
    ax[1].set_title("V{}_sig".format(component))
    ax[1].set_xlim((west, east))
    ax[1].set_ylim((south, north))
    # ascending.plot(ax=ax[1], facecolor='None', edgecolor='red', alpha=1)
    # descending.plot(ax=ax[1], facecolor='None', edgecolor='blue', alpha=1)

    plt.savefig(f'{out_dir}/kriging_interpolation_v{component}.png', dpi=300, bbox_inches='tight')
    plt.close()


def monte_carlo_interpolation(gps, num):
    """
    Monte Carlo method to calculate num of independently interpolated velocities
    from randomly perturbed GNSS velocities
    @param gps: gps geopandas dataframe
    @param num: number of independent interpolation you wish to generate
    @return: None, but interpolated field saved to NETCDF grid files
    """
    for i in np.arange(num):
        comp = "vel_monte"
        # 1. perturb gps velocity by a random amount sampled from a normal distribution with sigma(v)
        gps[comp] = gps[vel].to_numpy() + [np.random.normal(0, sigma, 1)[0] for sigma in gps[sig]]
        # 2. kriging
        vel_interpolated, sig_interpolated = grid_gps(gps, gridx, gridy, comp)
        # 3. plot
        plot_interpolation(vel_interpolated, sig_interpolated)
        # 4. write data to file
        out_dir = "../kriging_monte_carlo/with_external_drift/"
        write_asc_grid(gridx, gridy, vel_interpolated, filename=out_dir+"v{}_interpolated_monte_carlo_{}.grd".format(component, i))


def load_frames(frame_file):
    """ load kml frames into a polygon geopandas dataframe
    Only for plotting in this script, not essential """
    frames = gpd.read_file(frame_file, driver='KML')
    frames['track'] = frames['Name'].str[:4]
    frames['orientation'] = frames['Name'].str[3]
    tracks = frames.dissolve(by='track', aggfunc='sum')
    tracks['orientation'] = tracks['Name'].str[3]
    dsc = tracks[tracks['orientation'] == 'D']
    asc = tracks[tracks['orientation'] == 'A']
    ascending = asc.dissolve(by='orientation', aggfunc='sum')
    descending = dsc.dissolve(by='orientation', aggfunc='sum')
    return ascending, descending



if __name__ == "__main__":
    # ----------------------------
    # Parse user arguments
    # ----------------------------
    parser = argparse.ArgumentParser(description="GPS Kriging with PyKrige")

    parser.add_argument("--region", type=float, nargs=4, metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                        default=[31.5, 45, 32, 43],
                        help="Region bounding box: lon_min lon_max lat_min lat_max")

    parser.add_argument("--2d_path", type=str, required=True,
                        help="Path to GNSS AHB 2D data file")

    parser.add_argument("--3d_path", type=str, required=True,
                        help="Path to GNSS AHB 3D data file")

    parser.add_argument("--res", type=float, default=0.2,
                        help="Grid resolution in degrees")

    parser.add_argument("--out_dir", type=str, default="gps_kriging",
                        help="Output directory name")
    parser.add_argument("--method", type=str, choices=["external_drift", "detrended", "specified_drift"],
                        default="external_drift",
                        help="Kriging interpolation method to use (default: external_drift)")


    args = parser.parse_args()

    # ----------------------------
    # Assign inputs from args
    # ----------------------------
    lon_min, lon_max, lat_min, lat_max = args.region
    ahb_2d_path = args.__dict__['2d_path']
    ahb_3d_path = args.__dict__['3d_path']
    res = args.res
    method = args.method
    
    # Final output directory includes method
    out_dir = os.path.join(args.out_dir, method)
    os.makedirs(out_dir, exist_ok=True)

    # For plotting and grid
    west, east, south, north = lon_min, lon_max, lat_min, lat_max

    # ----------------------------
    # Load GPS data
    # ----------------------------
    print("Loading GPS and performing data culling ...")
    gps = load_and_prepare_gps(ahb_2d_path, ahb_3d_path, lon_min, lon_max, lat_min, lat_max)

    # ----------------------------
    # Prepare velocity component
    # ----------------------------
    component = 'n'
    vel = f'v{component}'
    sig = f's{component}'

    vel_vmin = gps[vel].quantile(0.01)
    vel_vmax = gps[vel].quantile(0.99)
    sig_vmin = 0
    sig_vmax = 2


    # ----------------------------
    # Grid generation
    # ----------------------------
    print("Kriging starts ... ")
    gridx = np.arange(west, east + res, res)
    gridy = np.arange(south, north + res, res)

    vel_interpolated, sig_interpolated = grid_gps(gps, gridx, gridy, vel, out_dir)

    # ----------------------------
    # Plot results
    # ----------------------------
    print("Plot Kriging Results ... ")
    plot_interpolation(vel_interpolated, sig_interpolated, out_dir)

    # ----------------------------
    # Save results
    # ----------------------------
    print("Saving Kriging Results ... ")
    write_asc_grid(gridx, gridy, vel_interpolated, filename=f"{out_dir}/{vel}_interpolated.grd")
    write_asc_grid(gridx, gridy, sig_interpolated, filename=f"{out_dir}/{sig}_interpolated.grd")
    

    # ----------------------------
    # Upsample and convert to GeoTIFF
    # ----------------------------

    print("Converting .grd to GeoTIFF and upsampling both velocity and sigma...")

    # File paths
    vel_grd = f"{out_dir}/{vel}_interpolated.grd"
    sig_grd = f"{out_dir}/{sig}_interpolated.grd"

    vel_tif = f"{out_dir}/{vel}_interpolated.tif"
    sig_tif = f"{out_dir}/{sig}_interpolated.tif"

    vel_tif_upsampled = f"{out_dir}/{vel}_interpolated_upsampled.tif"
    sig_tif_upsampled = f"{out_dir}/{sig}_interpolated_upsampled.tif"

    # Convert .grd to .tif
    subprocess.run(["gdal_translate", "-of", "GTiff", vel_grd, vel_tif], check=True)
    subprocess.run(["gdal_translate", "-of", "GTiff", sig_grd, sig_tif], check=True)

    # Upsample .tif to 0.01 deg resolution using bilinear interpolation
    subprocess.run([
        "gdalwarp",
        "-tr", "0.01", "0.01",
        "-r", "bilinear",
        vel_tif,
        vel_tif_upsampled
    ], check=True)

    subprocess.run([
        "gdalwarp",
        "-tr", "0.01", "0.01",
        "-r", "bilinear",
        sig_tif,
        sig_tif_upsampled
    ], check=True)

    print("Conversion and upsampling completed successfully.")


    # #################################################
    # #   For evaluating uncertainties in             #
    # #   gradients of interpolated velocity field    #
    # #################################################

    # # # monte carlo  - generate many interpolated fields for uncertainty analysis which I later performed in GMT
    # # print("Running Monte Carlo to interpolate from many perturbed GPS ... ")
    # # monte_carlo_interpolation(gps, 100)
