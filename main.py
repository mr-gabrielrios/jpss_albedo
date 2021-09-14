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
from scipy import spatial
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
import sklearn.metrics

import scripts.mesonet as mesonet
import scripts.ameriflux_reader as ameriflux
import scripts.viirs_reader as viirs_reader

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
    tree = spatial.KDTree(coords)
    dist, coord_idx = tree.query([(target_lat, target_lon)])
    # Get coordinate 'hit' given the queried coordinate
    hit_lat, hit_lon = coords[coord_idx[0]]
    print(target_lat, target_lon, ' | ', hit_lat, hit_lon, ' | ', dist)
    
    # Determine the indices that correspond to the 'hit' coordinate
    lat_idx = list(zip(*np.where(lat == hit_lat)))
    lon_idx = list(zip(*np.where(lon == hit_lon)))
    # Gather indices that correspond to the 'hit' coordinate and ensure unique index pair
    if set(lat_idx) & set(lon_idx):
        nearest_data = data[np.where(lon == hit_lon)[0][0], np.where(lon == hit_lon)[1][0]]
    else:
        nearest_data = np.nan
    
    return nearest_data, hit_lat, hit_lon, dist

def viirs(dirpath, loc, date_range, target_lat, target_lon, box_size):
    
    ''' Processes VIIRS satellite data given a directory, location, and target coordinate. '''
    
    # Get list of datasets that correspond to the target coordinate
    datasets = viirs_reader.main(dirpath, date_range, target_lat, target_lon, box_size)
    # Initialize empty dictionary to hold dataset information
    # This dictionary will be used for future concatenation
    data = {}
    # For every dataset, gather the albedo value closest to the target coordinate
    for i, dataset in enumerate(datasets):
        print('{0} of {1} VIIRS datasets completed'.format(i+1, len(datasets)))
        val, lat, lon, dist = find_nearest(dataset['Latitude'].data,
                               dataset['Longitude'].data,
                               dataset['VIIRS_Albedo_EDR'].data,
                               target_lat, target_lon)
        # Multiply albedo value by a scale value for normalization
        surfalb = val * dataset['AlbScl'].mean().values
        datetime = dataset.time_coverage_start
        # Append data to the dictionary for future concatenation
        data[i] = {'datetime': datetime, 'lat': lat, 'lon': lon, 'surfalb': surfalb, 'loc': loc, 'dist': dist}
        
    data = pd.DataFrame.from_dict(data, orient='index')
        
    return data

def mbe(data, site):
    
    ''' Calculate mean bias error given a DataFrame with 'alb_viirs' and 'alb_obs' columns and a location. '''
    
    y_true = data.loc[data['loc'] == site]['alb_obs'].to_numpy()
    y_pred = data.loc[data['loc'] == site]['alb_viirs'].to_numpy()
    y_true = y_true.reshape(len(y_true),1)
    y_pred = y_pred.reshape(len(y_pred),1)   
    diff = (y_true-y_pred)
    mbe = diff.mean()
    
    return mbe

def rmse(data, site):
    
    ''' Calculate root mean square error given a DataFrame with 'alb_viirs' and 'alb_obs' columns and a location. '''
    
    y_true = data.loc[data['loc'] == site]['alb_obs'].to_numpy()
    y_pred = data.loc[data['loc'] == site]['alb_viirs'].to_numpy()
    y_true = y_true.reshape(len(y_true),1)
    y_pred = y_pred.reshape(len(y_pred),1)   
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_true, y_pred))
    
    return rmse

def main(dirpath, locs, date_range, box_size):
    
    ''' Gather VIIRS and field observation data. '''
    
    viirs_data, mesonet_data, ameriflux_data = [], [], []
    mesonet_stations = ['BKLN', 'QUEE', 'STAT']
    ameriflux_stations = ['Ro6', 'IB1']
    for loc in locs:
        print('\n', loc)
        print('------------------------')
        # Assign coordinates for each Mesonet station (deg N, deg E)
        if loc == 'BKLN':
            target_lat, target_lon = [40.631762, -73.953678]
        elif loc == 'QUEE':
            target_lat, target_lon = [40.734335, -73.815856]
        elif loc == 'STAT':
            target_lat, target_lon = [40.604014, -74.148499]
        elif loc == 'Ro6':
            target_lat, target_lon = [44.6946, -93.0578]
        elif loc == 'IB1':
            target_lat, target_lon = [41.8593, -88.2227]
        
        # Append data to respective lists
        viirs_data.append(viirs(dirpath, loc, date_range, target_lat, target_lon, box_size))
        if loc in mesonet_stations:
            mesonet_data.append(mesonet.main(date_range, loc))
        else:
            ameriflux_data.append(ameriflux.processor(date_range, loc))
        
    # Concatenate DataFrames
    viirs_data = pd.concat(viirs_data)
    mesonet_data = pd.concat(mesonet_data)
    ameriflux_data = pd.concat(ameriflux_data)
        
    return viirs_data, mesonet_data, ameriflux_data

if __name__ == '__main__':
    # Define directory containing all surface albedo data
    dirpath = os.path.join(os.getcwd(), 'data', 'viirs')
    # Bound box size (in degrees)
    box_size = 2
    # Date range of interest
    date_range = [datetime.datetime(year=2019, month=6, day=1, hour=0),
                  datetime.datetime(year=2020, month=5, day=31, hour=0)-datetime.timedelta(hours=1)]
    # Locations of interest
    locs = ['BKLN', 'QUEE', 'STAT', 'Ro6', 'IB1']
    
    viirs_data, mesonet_data, ameriflux_data = main(dirpath, locs, date_range, box_size)
    
    viirs_data['dist'] = viirs_data['dist'].apply(lambda x: x[0])
    viirs_data.index = pd.to_datetime(viirs_data['datetime'])
    viirs_data_resampled = []
    mesonet_data_resampled = []
    ameriflux_data_resampled = []
    
    for name, group in viirs_data.groupby('loc'):
        temp = group.resample('1D').mean()
        temp['loc'] = name
        viirs_data_resampled.append(temp)
    viirs_data_resampled = pd.concat(viirs_data_resampled)
    
    for name, group in mesonet_data.groupby('loc'):
        temp = group.resample('1D').mean()
        temp['loc'] = name
        mesonet_data_resampled.append(temp)
    mesonet_data_resampled = pd.concat(mesonet_data_resampled)
    
    for name, group in ameriflux_data.groupby('loc'):
        temp = group.resample('1D').mean()
        temp['loc'] = name
        ameriflux_data_resampled.append(temp)
    ameriflux_data_resampled = pd.concat(ameriflux_data_resampled)
    
    # obs_data = mesonet_data_resampled.append(ameriflux_data_resampled, ignore_index=True)
    
    obs_data = pd.concat([mesonet_data_resampled, ameriflux_data_resampled])
    
    agg_data = pd.merge(viirs_data_resampled, obs_data, on=["datetime", "loc"])
    agg_data = agg_data.rename(columns={'surfalb_x': 'alb_viirs', 'surfalb_y': 'alb_obs'})
    mask = (agg_data['alb_obs'] < 0) | (agg_data['alb_obs'] > 1)
    agg_data = agg_data[~mask]
    
    agg_data_resampled = agg_data.groupby('loc').resample('8D').mean()
    mask = np.isnan(agg_data_resampled['alb_obs']) | np.isnan(agg_data_resampled['alb_viirs'])
    agg_data_resampled = agg_data_resampled[~mask]
    agg_data_resampled = agg_data_resampled.reset_index()
    agg_data_resampled['error'] = (agg_data_resampled['alb_obs'] - agg_data_resampled['alb_viirs'])/agg_data_resampled['alb_obs']
    r2 = sklearn.metrics.r2_score(agg_data_resampled['alb_obs'], agg_data_resampled['alb_viirs'])
    rmse = np.sqrt(sklearn.metrics.mean_squared_error(agg_data_resampled['alb_obs'], agg_data_resampled['alb_viirs']))
    
    # Performance plot
    fig, ax = plt.subplots(dpi=300)
    im = sns.scatterplot(data=agg_data_resampled, x='alb_obs', y='alb_viirs', hue='loc', ax=ax)
    im2 = ax.plot([0, 0.5], [0, 0.5], linestyle='--', color='black')
    ax.set_aspect('equal')
    ax.set_xlim([0, 0.5])
    ax.set_ylim([0, 0.5])
    
    '''
    # Satellite retrieval locations
    bound_box = [target_lon-0.5, target_lon+0.5, target_lat-0.5, target_lat+0.5]
    fig, ax = plt.subplots(dpi=144, subplot_kw={'projection': ccrs.PlateCarree()})
    im_viirs = sns.scatterplot(agg_data['lon_x'], agg_data['lat_x'])
    im_obs = sns.scatterplot(agg_data['lon_y'], agg_data['lat_y'])
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label('Surface albedo', rotation=270, labelpad=15)
    ax.set_extent(bound_box)
    ax.coastlines()
    '''