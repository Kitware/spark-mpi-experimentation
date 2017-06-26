from __future__ import print_function
import os
import sys
import time
import gdal
import numpy as np


# -------------------------------------------------------------------------
# Files to process
# -------------------------------------------------------------------------

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
basepath = '/data/sebastien/SparkMPI/data/gddp'

# -------------------------------------------------------------------------
# Read file and output (year|month, temp)
# -------------------------------------------------------------------------

def readFile(fileName):
    year = fileName.split('_')[-1][:-4]
    print('year', year)
    dataset = gdal.Open('%s/%s' % (basepath, fileName))
    for bandId in range(dataset.RasterCount):
        band = dataset.GetRasterBand(bandId + 1).ReadAsArray()
        for value in band.flatten():
            yield (year, value)


# -----------------------------------------------------------------------------

def readFileAndCompute(fileName):
    year = fileName.split('_')[-1][:-4]
    print('year', year)
    dataset = gdal.Open('%s/%s' % (basepath, fileName))
    total = 0
    count = 0
    for bandId in range(dataset.RasterCount):
        band = dataset.GetRasterBand(bandId + 1).ReadAsArray()
        for value in band.flatten():
            if value < 50000:
                total += value
                count += 1

    return (year, total / count)

# -----------------------------------------------------------------------------

def readDoNothing(fileName):
    year = fileName.split('_')[-1][:-4]
    print('year', year)
    dataset = gdal.Open('%s/%s' % (basepath, fileName))
    for bandId in range(dataset.RasterCount):
        band = dataset.GetRasterBand(bandId + 1).ReadAsArray()
        print(band.shape)

# -------------------------------------------------------------------------
# Read timing
# -------------------------------------------------------------------------

t0 = time.time()
for fileName in fileNames:
    readDoNothing(fileName)

t1 = time.time()
print('### Total execution time - %s ' % str(t1 - t0))

