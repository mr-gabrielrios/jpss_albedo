'''
Script name:        NYS Mesonet File Conversion - mesonet_converter.py
Script objective:   The objective of this script is to generate files to match the format 
                    given by NYS Mesonet for flux data 
'''

### Imports
import os, sys
from datetime import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

##########################################################
# File Conversion
# Objective:    Convert files into desired format and generate files accordingly
# Input:        directory name for files of interest (str)
# Output:       none

def file_conversion(fpath, locs):
    
    ## Date range setup
    # Assume that the data date range is in the directory name, with '_' being the separator
    start_date, end_date = ['20191001', '20200531']
    start_date, end_date = [datetime.strptime(start_date, '%Y%m%d'),
                            datetime.strptime(end_date, '%Y%m%d')+timedelta(days=1)]
    
    # Get every day between bounds of the time range
    date_range = pd.date_range(start_date, end_date, (end_date-start_date).total_seconds()/3600/24+1)
    date_range = [t.to_pydatetime() for t in date_range]
    
    ## Station list in NYS Mesonet network, New York City metro area only
    station_list = locs
    
    ## Column list for data reading
    # Station ID, datetime, sensible heat flux, friction velocity, air temperature
    data_cols = ['stid', 'datetime', 'H', 'USTAR', 'Tc', 'SW_IN', 'SW_OUT', 'LW_IN', 'LW_OUT', 'Rn'] 
    
    ## Filter 1: Create DataFrame with all data filtered by selected columns
    all_data = pd.DataFrame()
    for file in os.listdir(fpath):
        df = pd.read_csv(os.path.join(fpath, file), usecols=data_cols)
        all_data = pd.concat([all_data, df])
    
    # Note:     I know I should've used a dictionary to store different station DataFrames, but this is just \
    #           an auxiliary script I won't use often, if ever again.
    
    ## Filter 2: Create DataFrames with all data filtered by station
    for loc in locs:
        loc_df = pd.DataFrame()
        if loc == 'BKLN':
            loc_df = all_data[all_data['stid'] == 'FLUX_BKLN']
            loc_df['datetime'] = loc_df['datetime'].astype('str')
        elif loc == 'BRON':
            loc_df = all_data[all_data['stid'] == 'FLUX_BRON']
            loc_df['datetime'] = loc_df['datetime'].astype('str')
        elif loc == 'MANH':
            loc_df = all_data[all_data['stid'] == 'FLUX_MANH']
            loc_df['datetime'] = loc_df['datetime'].astype('str')
        elif loc == 'QUEE':
            loc_df = all_data[all_data['stid'] == 'FLUX_QUEE']
            loc_df['datetime'] = loc_df['datetime'].astype('str')
        elif loc == 'STAT':
            loc_df = all_data[all_data['stid'] == 'FLUX_STAT']
            loc_df['datetime'] = loc_df['datetime'].astype('str')
        else:
            print('Station not found! Ending script...')
            break
    
        # Calculate albedo from radiation data when net radiation is positive (sun is up)
        # Reference: Land Surface Albedo ATBD, Section 2.5.1
        #            Link: https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/ATBD_EPS_Land_SurfaceAlbedo_v1.3.pdf
        loc_df['ALB'] = np.where(loc_df['Rn'] > 0, loc_df['SW_OUT']/loc_df['SW_IN'], 0)
    
        data_fpath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data', 'vdtn', loc))
        
        # Make directory if it doesn't already exist
        if not os.path.isdir(data_fpath):
            os.mkdir(data_fpath)
        
        for day in date_range:
            fname = day.strftime('%Y%m%d') + '_' + 'FLUX_' + loc + '_Flux_NYSMesonet.csv'
            day = day.strftime('%Y-%m-%d')
            if not loc_df[loc_df['datetime'].str[:10] == day].empty:
                temp = loc_df[loc_df['datetime'].str[:10] == day]
                # Avoid re-writing copies of the same file
                temp.to_csv(path_or_buf=os.path.join(data_fpath, fname))
        print('{0} done'.format(loc))         
       
    return loc_df
            
fpath = '/Users/gabriel/Documents/noaa_cessrst/nerto/jpss_albedo/data/vdtn/mesonet_raw'
test_data  = file_conversion(fpath, ['BKLN', 'QUEE', 'STAT'])