#!/data/sebastien/SparkMPI/pv-install/bin/pvpython

# ----------------------------------------------------
# Testing script by copying network output to a file
# $ nc -l 65432 > net.txt
# ----------------------------------------------------

from __future__ import print_function
import os
import sys
import time
import gdal
import numpy as np
import socket

# -------------------------------------------------------------------------
# Network config
# -------------------------------------------------------------------------

hostname = 'localhost'
port = 65432
server = True

# -------------------------------------------------------------------------
# Files to process
# -------------------------------------------------------------------------

basepath = '/data/scott/SparkMPI/data/gddp'

fileNames = [
    'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2006.tif',
    'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2007.tif',
    'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2008.tif',
    'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2009.tif',
    'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2010.tif',
    'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2011.tif',
    'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2012.tif',
    'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2013.tif',
    'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2014.tif',
    'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2015.tif',
]

# -------------------------------------------------------------------------
# Read file and push data to socket
# -------------------------------------------------------------------------

def readFileData(fileName):
    year = fileName.split('_')[-1][:-4]
    print('year', year)
    dataset = gdal.Open('%s/%s' % (basepath, fileName))
    for bandId in range(dataset.RasterCount):
        band = dataset.GetRasterBand(bandId + 1).ReadAsArray()
        print(' - band: %d' % bandId)
        for value in band.flatten():
            yield value

# -------------------------------------------------------------------------
# Read timing
# -------------------------------------------------------------------------

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
channel = None

if server:
    s.bind((hostname, port))
    s.listen(1)
    conn, addr = s.accept()
    channel = conn
else:
    s.connect((hostname, port))
    channel = s

t0 = time.time()
for fileName in fileNames:
    for value in readFileData(fileName):
        channel.send('%s\n' % str(value))

t1 = time.time()
print('### Total execution time - %s ' % str(t1 - t0))


