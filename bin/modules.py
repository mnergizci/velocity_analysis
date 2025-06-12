#!/usr/bin/env python3

import os
from osgeo import gdal
import numpy as np
import pandas as pd
import shapely.speedups
from cmcrameri import cm
from matplotlib import pyplot as plt
from scipy.stats import mode
gdal.UseExceptions()
from fiona.drvsupport import supported_drivers
supported_drivers['LIBKML'] = 'rw'
# gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
shapely.speedups.enable()


class OpenTif:
    """ a Class that stores the band array and metadata of a Gtiff file.
    Can be initiated using a Tiff file alone, or with additional bands from
    its associated uncertainties, incidence and heading files. """
    def __init__(self, filename, sigfile=None, incidence=None, heading=None, N=None, E=None, U=None):
        self.ds = gdal.Open(filename)
        self.basename = os.path.splitext(os.path.basename(filename))[0]
        self.band = self.ds.GetRasterBand(1)
        self.data = self.band.ReadAsArray()
        self.xsize = self.ds.RasterXSize
        self.ysize = self.ds.RasterYSize
        self.left = self.ds.GetGeoTransform()[0]
        self.top = self.ds.GetGeoTransform()[3]
        self.xres = self.ds.GetGeoTransform()[1]
        self.yres = self.ds.GetGeoTransform()[5]
        self.right = self.left + self.xsize * self.xres
        self.bottom = self.top + self.ysize * self.yres
        self.projection = self.ds.GetProjection()
        pix_lin, pix_col = np.indices((self.ds.RasterYSize, self.ds.RasterXSize))
        self.lat, self.lon = self.top + self.yres*pix_lin, self.left+self.xres*pix_col

        # convert 0 and 255 to NaN
        #self.data[self.data == 0.00000000000000] = np.nan
        self.data[self.data == 255] = np.nan

        if sigfile is not None:
            self.dst = gdal.Open(sigfile)
            self.bandt = self.dst.GetRasterBand(1)
            self.sigma = self.bandt.ReadAsArray()
            self.sigma[self.sigma == 0] = np.nan
            if self.dst.RasterXSize != self.xsize or self.dst.RasterYSize != self.ysize:
                try:
                    self.sigma = self.sigma[:self.ysize, :self.xsize]
                except Warning:
                    print('Error: Sigma and Velocity file not the same size!')
                    print('sig has size = ' + str(self.dst.RasterXSize) + ', ' + str(self.dst.RasterYSize))
                    print('vel has size = ' + str(self.ds.RasterXSize) + ', ' + str(self.ds.RasterYSize))
                # self.sigma = np.ones((self.ysize, self.xsize))
        else:
            self.sigma = np.ones((self.ysize, self.xsize))

        if incidence is not None:
            self.ds_inc = gdal.Open(incidence)
            self.band_inc = self.ds_inc.GetRasterBand(1)
            self.inc = np.deg2rad(self.band_inc.ReadAsArray())
            self.inc[self.inc == 0] = np.nan
            if self.ds_inc.RasterXSize != self.xsize or self.ds_inc.RasterYSize != self.ysize:
                try:
                    self.inc = self.inc[:self.ysize, :self.xsize]
                except Warning:
                    print('Error: Inc and Velocity file not the same size!')
                    print('inc has size = ' + str(self.ds_inc.RasterXSize) + ', ' + str(self.ds_inc.RasterYSize))
                    print('vel has size = ' + str(self.ds.RasterXSize) + ', ' + str(self.ds.RasterYSize))

        if heading is not None:
            self.ds_head = gdal.Open(heading)
            self.band_head = self.ds_head.GetRasterBand(1)
            self.head = np.deg2rad(self.band_head.ReadAsArray())
            self.head[self.head == 0] = np.nan
            if self.ds_head.RasterXSize != self.xsize or self.ds_head.RasterYSize != self.ysize:
                try:
                    self.head = self.head[:self.ysize, :self.xsize]
                except Warning:
                    print('Error: Heading and Velocity file not the same size!')
                    print('head has size = ' + str(self.ds_head.RasterXSize) + ', ' + str(self.ds_head.RasterYSize))
                    print('vel has size = ' + str(self.ds.RasterXSize) + ', ' + str(self.ds.RasterYSize))

        if N is not None:
            self.ds_N = gdal.Open(N)
            self.band_N = self.ds_N.GetRasterBand(1)
            self.N = self.band_N.ReadAsArray()
            # self.N[self.N == 0] = np.nan
            if self.ds_N.RasterXSize != self.xsize or self.ds_N.RasterYSize != self.ysize:
                try:
                    self.N = self.N[:self.ysize, :self.xsize]
                except Warning:
                    print('Error: Heading and Velocity file not the same size!')
                    print('head has size = ' + str(self.ds_N.RasterXSize) + ', ' + str(self.ds_N.RasterYSize))
                    print('vel has size = ' + str(self.ds.RasterXSize) + ', ' + str(self.ds.RasterYSize))

        if E is not None:
            self.ds_E = gdal.Open(E)
            self.band_E = self.ds_E.GetRasterBand(1)
            self.E = self.band_E.ReadAsArray()
            # self.E[self.E == 0] = np.nan
            if self.ds_E.RasterXSize != self.xsize or self.ds_E.RasterYSize != self.ysize:
                try:
                    self.E = self.E[:self.ysize, :self.xsize]
                except Warning:
                    print('Error: Heading and Velocity file not the same size!')
                    print('head has size = ' + str(self.ds_E.RasterXSize) + ', ' + str(self.ds_E.RasterYSize))
                    print('vel has size = ' + str(self.ds.RasterXSize) + ', ' + str(self.ds.RasterYSize))

        if U is not None:
            self.ds_U = gdal.Open(U)
            self.band_U = self.ds_U.GetRasterBand(1)
            self.U = self.band_U.ReadAsArray()
            # self.U[self.U == 0] = np.nan
            if self.ds_U.RasterXSize != self.xsize or self.ds_U.RasterYSize != self.ysize:
                try:
                    self.U = self.U[:self.ysize, :self.xsize]
                except Warning:
                    print('Error: Heading and Velocity file not the same size!')
                    print('head has size = ' + str(self.ds_U.RasterXSize) + ', ' + str(self.ds_U.RasterYSize))
                    print('vel has size = ' + str(self.ds.RasterXSize) + ', ' + str(self.ds.RasterYSize))

    def extract_pixel_value(self, lon, lat, max_width=200):
        """ Extract pixel values from a geo raster at given lat lon locations
        by searching for the smallest window with enough non-empty pixels to
        calculate an average and standard deviation values.
        max_width is the upper bound of the size of the search window.
        If there are not enough pixels with values in the window, it is considered that
        there is no matching value from the raster. """
        x = int((lon-self.left)/self.xres+0.5)
        y = int((lat - self.top) / self.yres + 0.5)
        # increase window size in steps of 2 until there are non-nan values in the window
        # starting from 2 with 5x5 window because if 1x1 window, stdev will be zero
        # if we use the std of values instead of the corresponding sigma files as stdev
        for n in np.arange(2, max_width+1, 2):
            pixel_values = self.data[y - n: y + n + 1, x - n: x + n + 1]
            index = np.nonzero(~np.isnan(pixel_values))
            if len(index[0]) > 10:
                # print(n, pixel_values)
                break
        if len(pixel_values) == 0 or np.isnan(pixel_values).all():
            pixel_value = np.nan
            stdev = np.nan
        else:
            pixel_value = np.nanmean(pixel_values)
            stdev = np.nanstd(pixel_values)
        return pixel_value, stdev

    def extract_pixel_value2(self, lon, lat, max_width=200):
        """ Extract pixel values from a geo raster at given lat lon locations
        by searching for the smallest window with enough non-empty pixels to
        calculate an average and standard deviation values.
        max_width is the upper bound of the size of the search window.
        If there are not enough pixels with values in the window, it is considered that
        there is no matching value from the raster. """
        x = int((lon-self.left)/self.xres+0.5)
        y = int((lat - self.top) / self.yres + 0.5)
        # increase window size in steps of 2 until there are non-nan values in the window
        # starting from 2 with 5x5 window because if 1x1 window, stdev will be zero
        # if we use the std of values instead of the corresponding sigma files as stdev
        for n in np.arange(2, max_width+1, 2):
            pixel_values = self.data[y - n: y + n + 1, x - n: x + n + 1]
            index = np.nonzero(~np.isnan(pixel_values))
            if len(index[0]) > 10:
                # print(n, pixel_values)
                break
        if len(pixel_values) == 0 or np.isnan(pixel_values).all():
            pixel_value = np.nan
            stdev = np.nan
        else:
            pixel_value = np.nanmedian(pixel_values)
            stdev = np.nanstd(pixel_values)
        return pixel_value, stdev

    def extract_inc(self, lon, lat):
        x = int((lon-self.left)/self.xres+0.5)
        y = int((lat - self.top) / self.yres + 0.5)
        inc = self.inc[y, x]
        return inc

    def extract_head(self, lon, lat):
        x = int((lon-self.left)/self.xres+0.5)
        y = int((lat - self.top) / self.yres + 0.5)
        head = self.head[y, x]
        return head

    def plot(self):
        plt.imshow(self.data, vmin=np.nanpercentile(self.data, 1), vmax=np.nanpercentile(self.data, 99), cmap=cm.roma.reversed())
        plt.colorbar()
        plt.title(self.basename)
        plt.show()


def export_tif(data, df, filename):
    # Export data to tif format.
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(filename, df.xsize, df.ysize, 1, gdal.GDT_Float32)
    outdata.SetGeoTransform([df.left, df.xres, 0, df.top, 0, df.yres])  ##sets same geotransform as input
    # outdata.SetProjection(df.projection)  ##sets same projection as input
    outdata.GetRasterBand(1).WriteArray(data)
    outdata.FlushCache()
    outdata.FlushCache()  # need to flush twice to export the last tif properly, otherwise it stops halfway.


def non_nan_merge(big_data, small_data, nodata_test, x_shift, y_shift, xsize, ysize):
    masked_data = np.choose(nodata_test,  # False = not nan = pick from 0th entry; True = nan = pick from 1st entry
        (small_data, big_data[y_shift:y_shift+ysize, x_shift:x_shift+xsize]))
    big_data[y_shift:y_shift+ysize, x_shift:x_shift+xsize] = masked_data


class Canvas(object):
    """ a Class that stores the dimensional specification of a grid mesh."""
    def __init__(self, north=None, south=None, west=None, east=None, x_step=None, y_step=None, width=None, length=None):
        if west is not None:
            self.left = west
        if north is not None:
            self.top = north
        if east is not None:
            self.right = east
        if south is not None:
            self.bottom = south
        if x_step is not None:
            self.xres = x_step
        if y_step is not None:
            self.yres = y_step
        if width is not None:
            self.xsize = width
        else:
            self.xsize = int((self.right - self.left) / self.xres + 1.5)
        if length is not None:
            self.ysize = length
        else:
            self.ysize = int((self.bottom - self.top) / self.yres + 1.5)
            
    # Define the display_info method
    def display_info(self):
        print(f"Canvas boundaries: North={self.top}, South={self.bottom}, West={self.left}, East={self.right}")
        print(f"Resolution: X step = {self.xres}, Y step = {self.yres}")
        print(f"Grid size: X size = {self.xsize} (columns), Y size = {self.ysize} (rows)")

def choose_3d_gps(gps_df, number_limit):
    """
    Only use 3D gps if there are enough of them.
    If the number of 3D gps is below threshold,
    use 2D stations with Vu = 0, and Su = the largest absolute vu + largest su in the frame
    """
    gps_2D = gps_df[gps_df['vu'] == 0]
    gps_3D = gps_df[gps_df['vu'] != 0]
    if len(gps_3D) > 0:
        if len(gps_3D) < number_limit:
            if gps_3D['vu'].max() + gps_3D['vu'].min() > 0:
                max_absolute_vu = gps_3D['vu'].max()
            else:
                max_absolute_vu = -gps_3D['vu'].min()
            gps_2D.loc[:, 'su'] = max_absolute_vu
            gps = pd.concat([gps_2D, gps_3D])
        else:
            gps = gps_3D
    else:
        gps = gps_2D
    return gps


def add_NEU_los(gps_df, N_coef_map, E_coef_map, U_coef_map):
    """calculate GPS LOS at gps lon lat from NEU and GPS component velocities"""
    # extract NEU values at gps lat lon from the NEU maps
    gps_df.loc[:, 'n_coef'] = [N_coef_map.extract_pixel_value(point.x, point.y)[0] for point in gps_df['geometry']]
    gps_df.loc[:, 'e_coef'] = [E_coef_map.extract_pixel_value(point.x, point.y)[0] for point in gps_df['geometry']]
    gps_df.loc[:, 'u_coef'] = [U_coef_map.extract_pixel_value(point.x, point.y)[0] for point in gps_df['geometry']]
    # calculate los and los uncertainty per gps point
    gps_df.loc[:, 'los'] = [ vn * n + ve * e + vu * u for vn, ve, vu, n, e, u
                            in gps_df[['vn', 've', 'vu', 'n_coef', 'e_coef', 'u_coef']].to_numpy()]
    gps_df.loc[:, 'los_sigma'] = [ abs(sn * n) + abs(se * e) + abs(su * u) for sn, se, su, n, e, u
                            in gps_df[['sn', 'se', 'su', 'n_coef', 'e_coef', 'u_coef']].to_numpy()]


def neu2los(vn, ve, vu, theta, phi):
    """ Calculating LOS velocities from N, E, U component velocities based on heading and incidence angles"""
    los = - ve * np.cos(phi * np.pi / 180) * np.sin(theta * np.pi / 180) \
          + vn * np.sin(phi * np.pi / 180) * np.sin(theta * np.pi / 180) \
          + vu * np.cos(theta * np.pi / 180)
    return los


def neu2los_sig(vn, ve, vu, theta, phi):
    """ Calculating uncertaintiges of LOS velocities from uncertainties of N, E, U component velocities
    based on heading and incidence angles"""
    los = np.sqrt( \
            ( ve * np.cos(phi * np.pi / 180) * np.sin(theta * np.pi / 180) ) ** 2 \
          + ( vn * np.sin(phi * np.pi / 180) * np.sin(theta * np.pi / 180) ) ** 2 \
          + ( vu * np.cos(theta * np.pi / 180) ) ** 2 )
    return los
