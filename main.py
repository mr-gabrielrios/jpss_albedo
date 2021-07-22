"""
NOAA-CESSRST NERTO Albedo Study
Script:         Main
Objective:      Compiles functions from all scripts in package to perform intended functions.
Developed by:   Gabriel Rios
"""

import datetime
import numpy as np
import os
import pandas as pd
import xarray as xr
import scipy

import bin.mesonet as mesonet
import bin.viirs_reader as viirs_reader

# Note: Albedo calculated by Kipp & Zonen CNR4 Radiometer
#       Manufacturer link: 
#       https://www.kippzonen.com/Download/354/Manual-CNR-4-Net-Radiometer-English-V2104

def find_nearest(lat, lon, data, target_lat, target_lon):
    
    ''' 
    Finds the nearest satellite pixel given 2D Numpy arrays of longitudes, latitudes, product data and two float values (latitude and longitude).
    '''
    
    # Get coordinate pairs in a list of ordered pairs (eliminate nans)
    coords = np.stack((np.ravel(lat), np.ravel(lon)), axis=-1)
    coords = coords[~np.isnan(coords).any(axis=1)]
    coords = list(map(tuple, coords))
    # Get KDTree to find the nearest neighbor
    tree = scipy.spatial.KDTree(coords)
    dist, coord_idx = tree.query([(target_lat, target_lon)])
    # Get coordinate 'hit' given the queried coordinate
    hit_lat, hit_lon = coords[coord_idx[0]]
    
    # Determine the indices that correspond to the 'hit' coordinate
    lat_idx = list(zip(*np.where(lat == hit_lat)))
    lon_idx = list(zip(*np.where(lon == hit_lon)))
    # Gather indices that correspond to the 'hit' coordinate and ensure unique index pair
    if set(lat_idx) & set(lon_idx):
        nearest_data = data[np.where(lon == hit_lon)[0][0], np.where(lon == hit_lon)[1][0]]
    else:
        nearest_data = np.nan
    
    return nearest_data, hit_lat, hit_lon

def viirs(dirpath, loc, target_lat, target_lon, box_size):
    
    ''' Processes VIIRS satellite data given a directory, location, and target coordinate. '''
    
    # Get list of datasets that correspond to the target coordinate
    datasets = viirs_reader.main(dirpath, target_lat, target_lon, box_size)
    # Initialize empty dictionary to hold dataset information
    # This dictionary will be used for future concatenation
    data = {}
    # For every dataset, gather the albedo value closest to the target coordinate
    for i, dataset in enumerate(datasets):
        print('{0} of {1} VIIRS datasets completed'.format(i+1, len(datasets)))
        val, lat, lon = find_nearest(dataset['Latitude'].data,
                               dataset['Longitude'].data,
                               dataset['VIIRS_Albedo_EDR'].data,
                               target_lat, target_lon)
        # Multiply albedo value by a scale value for normalization
        surfalb = val * dataset['AlbScl'].mean().values
        datetime = dataset.time_coverage_start
        # Append data to the dictionary for future concatenation
        data[i] = {'datetime': datetime, 'lat': lat, 'lon': lon, 'surfalb': surfalb, 'loc': loc}
        
    data = pd.DataFrame.from_dict(data, orient='index')
        
    return data

def main(dirpath, locs, date_range, box_size):
    
    ''' Gather VIIRS and field observation data. '''
    viirs_data, mesonet_data = [], []
    for loc in locs:
        # Assign coordinates for each Mesonet station (deg N, deg E)
        if loc == 'BKLN':
            target_lat, target_lon = [40.3605, -73.9521]
        elif loc == 'QUEE':
            target_lat, target_lon = [40.7366, -73.8201]
        elif loc == 'STAT':
            target_lat, target_lon = [40.6021, -74.1504]
        
        # Append data to respective lists
        viirs_data.append(viirs(dirpath, loc, target_lat, target_lon, box_size))
        mesonet_data.append(mesonet.main(date_range, loc))
        
    # Concatenate DataFrames
    viirs_data = pd.concat(viirs_data)
    mesonet_data = pd.concat(mesonet_data)
        
    return viirs_data, mesonet_data

if __name__ == '__main__':
    # Define directory containing all surface albedo data
    dirpath = os.path.join(os.getcwd(), 'data', 'viirs')
    # Coordinate of interest (latitude, longitude)
    target_lat, target_lon = [40.7128, -74.0060]
    # Bound box size (in degrees)
    box_size = 2
    # Date range of interest
    date_range = [datetime.datetime(year=2019, month=6, day=1, hour=0),
                  datetime.datetime(year=2019, month=9, day=1, hour=0)-datetime.timedelta(hours=1)]
    # Locations of interest
    locs = ['BKLN', 'QUEE', 'STAT']
    
    viirs_data, mesonet_data = main(dirpath, locs, date_range, box_size)