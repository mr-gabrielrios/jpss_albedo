"""
NOAA-CESSRST NERTO Albedo Study
Script:         VIIRS Data Tarball Handler
Objective:      Read list of .tar files output by NOAA-CLASS and dump directories with netCDF data into specified data folder.
Developed by:   Gabriel Rios
"""

import tarfile, os, sys

# Specified data folder to hold netCDF4 directories
dirpath = r'/Users/gabriel/Documents/noaa_cessrst/nerto/jpss_albedo/data/viirs'

# Assume .tar files are in data directory ('dir_path') specified above
# Iterate through all files in the target directory
for file in os.listdir(dirpath):
    # Identify .tar files
    if file.endswith(".tar"):
        # Generate absolute file path for the iterand .tar file
        fpath = os.path.join(dirpath, file)
        # Open .tar file
        tar = tarfile.open(fpath)
        # Get all files within the .tar file
        for member in tar.getmembers():
            # Generate name for the new directory where files from this .tar will be kept
            tardir = os.path.join(dirpath, file.split('.')[0])
            # If this directory doesn't exist, make it
            if not os.path.isdir(tardir):
                os.mkdir(tardir)
            # Extract files within .tar to the specified directory
            tar.makefile(member, os.path.join(tardir, member.name))
        tar.close()