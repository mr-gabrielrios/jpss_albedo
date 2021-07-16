#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NOAA-CESSRST NERTO Albedo Study
Script:         VIIRS Reader
Objective:      Read and process VIIRS data for a given coordinate.
Developed by:   Gabriel Rios
"""

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import os
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

def file_grab(dirpath, lat, lon):
    '''
    Return list of files within given parent directory that contain the defined target coordinate.
    '''
    
    # Initialize list of .nc files
    file_list, ds_list = [], []
    # Walk through directories
    for directory in os.listdir(dirpath):
        # Filter out Mac metadata
        if directory != '.DS_Store':
            # Print each file in the directory
            for file in os.listdir(os.path.join(dirpath, directory)):
                # Append file to list of files
                file_list.append(os.path.join(dirpath, directory, file))

    # For each file in the list, filter by identifying if given coordinate is in the array
    for file in sorted(file_list):
        ds = xr.open_dataset(file)
        check = coord_idx(ds, lat, lon)
        # If location in Dataset, collect Dataset
        if check:
            print(file)
            ds_list.append(xr.open_dataset(file))

    return ds_list

def coord_idx(dataset, lat, lon):
    '''
    Checks to see if the current Dataset contains the target coordinate.
    '''
    
    n = 1
    lat = np.around(lat, n)
    lon = np.around(lon, n)
    
    # Round latitude and longitude arrays to 'n' decimal places to allow for an approximate match
    lats = np.around(dataset['Latitude'].data, n)
    lons = np.around(dataset['Longitude'].data, n)
    # Get index of matching latitude
    lat_idx = np.where(lats == lat)
    # Check to see if matches exist. If not, return False. Else, check longitudes.
    if lat_idx[0].size == 0:
        return False
    else:
        # Grab corresponding longitudes
        lons_corr = lons[lat_idx]
        # If longitudes in corresponding indices, return True. Else, return False.
        if lon in lons_corr:
            return True
        else:
            return False
 
def grid_gen(ds, target_lat, target_lon, box_size, step):
    '''
    Creates coordinate grid in a square extent given a target latitude, target longitude,
    box size, and grid step size.
    '''
    
    grid_lons = np.arange(target_lon-box_size, target_lon+box_size, step)
    grid_lats = np.arange(target_lat-box_size, target_lat+box_size, step) 
 
def quick_test(ncdata, target_lat, target_lon, date):
    '''
    Prints a plot of the data on hand as a quick visualization of the working data.
    '''
     
    # Get latitude
    lat = ncdata['Latitude'][:].data
    # Get longitude
    lon = ncdata['Longitude'][:].data
    # Product of raw EDR albedo data and prescribed scale factor
    # Remove all values greater than 1 and bring to scale
    alb = np.where(ncdata['VIIRS_Albedo_EDR'].data >= 1/ncdata['AlbScl'].min().values, 
                   np.nan, ncdata['VIIRS_Albedo_EDR'].data)*ncdata['AlbScl'].mean().values
    pqi = ncdata['ProductQualityInformation'].data 
    
    # Filter by spatial extent box
    bound_box = [target_lon-0.5, target_lon+0.5, target_lat-0.5, target_lat+0.5]
    fig, ax = plt.subplots(dpi=144, subplot_kw={'projection': ccrs.PlateCarree()})
    im = ax.pcolormesh(lon, lat, alb, cmap='viridis', vmin=0, vmax=0.30)
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label('Surface albedo', rotation=270, labelpad=15)
    ax.set_extent(bound_box)
    ax.set_title('VIIRS LSA @ {0}'.format(date))
    
    
    return lat, lon

def main(dirpath, lat, lon, box_size):
    # Return file list
    ds_list = file_grab(dirpath, lat, lon)
    # Run through all location-filtered Datasets to ensure location check works properly
    for ds in ds_list:
        lat, lon = quick_test(ds, 40.7128, -74.0060, ds.time_coverage_start)
        
if __name__ == '__main__':
    ''' User inputs. '''
    # Define directory containing all surface albedo data
    dirpath = os.path.join(os.getcwd(), 'data', 'viirs')
    # Coordinate of interest (latitude, longitude)
    target_lat, target_lon = [40.7128, -74.0060]
    # Bound box size (in degrees)
    box_size = 2
    
    main(dirpath, target_lat, target_lon, box_size)