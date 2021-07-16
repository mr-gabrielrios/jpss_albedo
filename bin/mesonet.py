"""
NOAA-CESSRST NERTO Albedo Study
Script:         Mesonet Reader
Objective:      Read and process Mesonet data for a given coordinate.
Developed by:   Gabriel Rios
"""

##############################################################################################
# BEGIN IMPORTS
##############################################################################################

import datetime, numpy as np, os, pandas as pd, pytz
from pysolar.solar import *
from tzwhere import tzwhere

##############################################################################################
# END IMPORTS
##############################################################################################
  
##############################################################################################
# Method name:      file_sort
# Method objective: Set key for sorting files chronologically 
#                   (first 8 characters are date in YYYYMMDD format)
# Input(s):         fn [str]
# Outputs(s):       fn[0:8] [str]
##############################################################################################

def file_sort(fn):
    return(fn[0:8])

##############################################################################################
# Method name:      CtoK
# Method objective: Convert temperature in Celsius to Kelvin
# Input(s):         T [float]
# Outputs(s):       T + 273.15 [float]
##############################################################################################

def CtoK(T):
    return T + 273.15

##############################################################################################
# Method name:      csv_reader
# Method objective: Scrape relevant data from CSVs in a given directory for a temporal domain.
# Input(s):         T [float]
# Outputs(s):       T + 273.15 [float]
##############################################################################################

def csv_reader(date_range, data_dir):
    
    # Define CSV columns to extract data from
    cols = ['datetime', 'ALB', 'SW_IN', 'SW_OUT', 'LW_IN', 'LW_OUT']
    
    # Initialize empty DataFrame with predefined columns
    data = pd.DataFrame(columns=cols)
    
    # Sort files by date, assuming standard Mesonet filename convention
    # Example: YYYYMMDD_PARAMETER_LOCATION_Parameter_NYSMesonet.csv
    file_list = sorted(os.listdir(data_dir), key=file_sort)
    
    # Specify date range of interest
    # Format: YYYYMMDDHHMM
    date_range = pd.date_range(start=date_range[0], end=date_range[-1], freq='30min')
    # data['datetime'] = date_range
    
    # Iterate through sorted file list and extract data within date range with daily resolution
    for i, file in enumerate(file_list): 
        file_date = datetime.datetime.strptime(os.path.splitext(file)[0][0:8], '%Y%m%d')
        # Reduce datetimes to daily resolution to work on down-filtering
        days = [datetime.datetime.strptime(date_range[0].strftime('%Y%m%d'), '%Y%m%d'),
                datetime.datetime.strptime(date_range[-1].strftime('%Y%m%d'), '%Y%m%d')]
        # Filter files by day - any day within the date range will have a corresponding file
        if days[0] <= file_date <= days[-1]:
            filename = os.path.join(data_dir, file)
            data = data.append(pd.read_csv(filename, usecols=cols))
    
    # Convert date strings to datetime data type
    data['datetime'] = pd.to_datetime(data['datetime'])
    # Filter data entries by full datetime (includes hours and minutes)
    data = data[(data['datetime'] >= date_range[0]) & (data['datetime'] <= date_range[-1])] 
    data = data.reset_index(drop=True)
    
    # Account for missing observation data by inserting nans
    for i, date in enumerate(date_range):
        nanrow = pd.DataFrame([[np.nan] * len(cols)], columns=cols)
        nanrow['datetime'] = date
        if data.loc[i, ['datetime']].item() != date:
            data = pd.concat([data.iloc[:i], nanrow, data.iloc[i:]]).reset_index(drop=True)
        
    # Re-cast numerical strings as floats
    data['SURFALB'] = data['ALB'].astype(float)
    data['R_DOWN'] = data['SW_IN'].astype(float) + data['LW_IN'].astype(float)
    data['R_UP'] = data['SW_OUT'].astype(float) + data['LW_OUT'].astype(float)
    # Match parameter names to model parameter names
    data = data.drop(columns=['ALB', 'SW_IN', 'SW_OUT', 'LW_IN', 'LW_OUT'])
    # Make dates ~aware~
    data['datetime'] = [pytz.utc.localize(data.loc[idx, 'datetime']).to_pydatetime() for idx in range(len(data))]
    # Set datetime to index
    data = data.set_index('datetime')
    
    return data
    

def processor(data, locn, intv = '1D', param='SURFALB'):
    '''
    Processes albedo data for validation with VIIRS Land Surface Albedo data.
    Data processed per VIIRS Surface Albedo ATBD:
        https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/ATBD_EPS_Land_SurfaceAlbedo_v1.3.pdf
    '''
    
    # Assign coordinates for each Mesonet station (deg N, deg E)
    if locn == 'BKLN':
        lat, lon = [40.3605, -73.9521]
    elif locn == 'QUEE':
        lat, lon = [40.7366, -73.8201]
    elif locn == 'STAT':
        lat, lon = [40.6021, -74.1504]
    
    # Calculate solar zenith angle (rough estimate)
    data['SZA'] = [90 - get_altitude_fast(lat, lon, data.index.to_list()[idx]) for idx in range(len(data))]
    # Filter data based on solar zenith angle, per manufacturer suggestion
    #   Sec.: 1.1.6.4
    #   Link: https://www.kippzonen.com/Download/354/Manual-CNR-4-Net-Radiometer-English-V2104.pdf
    data.loc[data['SZA'] >= 80, 'SURFALB'] = 0
    
    # Set any negative or zero values to nan to preserve statistics
    data.loc[data['SURFALB'] <= 0, 'SURFALB'] = np.nan
    # # Generate empty DataFrame to calculate temporal statistics
    stats = pd.DataFrame()
    # Get mean over time interval
    stats['mean'] = data.resample(intv).mean()['SURFALB']
    # Get standard deviation over time interval
    stats['std'] = data.resample(intv).std()['SURFALB']
    # Get count
    stats['N'] = data.resample(intv).count()['SURFALB']
    
    return stats


if __name__ == "__main__":
    date_range = [datetime.datetime(year=2019, month=6, day=1, hour=0),
                  datetime.datetime(year=2019, month=10, day=1, hour=0)-datetime.timedelta(hours=1)]
    date_range = pd.date_range(start=date_range[0], end=date_range[1], freq='H') 
    data_dir = os.path.join(os.getcwd(), 'data/vdtn/BKLN/')
    data = csv_reader(date_range, data_dir)
    stats = processor(data, 'BKLN')