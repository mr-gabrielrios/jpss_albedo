"""
NOAA-CESSRST NERTO Albedo Study
Script:         Main
Objective:      Compiles functions from all scripts in package to perform intended functions.
Developed by:   Gabriel Rios
"""

import numpy as np
import os
import pandas as pd

import bin.mesonet as mesonet
import bin.viirs_reader as viirs_reader

# Note: Albedo calcualted by Kipp & Zonen CNR4 Radiometer
#       Manufacturer link: https://www.kippzonen.com/Download/354/Manual-CNR-4-Net-Radiometer-English-V2104

