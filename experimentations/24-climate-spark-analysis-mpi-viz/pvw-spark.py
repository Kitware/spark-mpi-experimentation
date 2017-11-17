from __future__ import print_function
import os
import sys
import time
import gdal
from datetime import datetime
import pyspark

from pyspark import SparkContext

from paraview import simple
import vtk
from paraview.vtk import vtkIOXML
import numpy as np
import numpy.ma as ma
import scipy.sparse as ss
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support
from vtk.vtkCommonCore import vtkIntArray


VALUE_RANGE = [0, 50000]


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
basepath = '/data/scott/SparkMPI/data/gddp'

# -------------------------------------------------------------------------
# Module-level variables and helper functions
# -------------------------------------------------------------------------

def swap(kv):
    return (kv[1], kv[0])

# -------------------------------------------------------------------------
#
# -------------------------------------------------------------------------

def readDay(dayFile):
    dayIdx = dayFile[1]
    fileName = dayFile[0]
    year = fileName.split('_')[-1][:-4]

    dataset = gdal.Open(fileName)
    imgDims = [dataset.RasterYSize, dataset.RasterXSize]
    numPixels = imgDims[0] * imgDims[1]

    result = []

    # for bandId in range(dataset.RasterCount):
    print('year: %s, band: %d' % (year, dayIdx + 1))
    band = ma.masked_outside(dataset.GetRasterBand(dayIdx + 1).ReadAsArray(), VALUE_RANGE[0], VALUE_RANGE[1])
    # .astype(np.dtype(np.int32))

    return (year, band)

def sumDays(accum, nextDay):
    return accum + nextDay

def average(data):
    year = data[0]
    print('Computing average for %s' % str(year))
    sumArray = data[1]
    return (year, sumArray / 365.0)

# -------------------------------------------------------------------------
# Parallel configuration
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# Partition handling
# -------------------------------------------------------------------------

def getMPIPartition(index):
    currentEdge = 0
    for i in range(len(processSlices)):
        currentEdge += (processSlices[i] * sliceSize)
        if index < currentEdge:
            return i

    return nbMPIPartition - 1


sc = SparkContext()

# -------------------------------------------------------------------------
# MPI configuration
# -------------------------------------------------------------------------



# -------------------------------------------------------------------------
# Spark pipeline
# -------------------------------------------------------------------------

# pathList will contain 3650 entries like ('/data/scott/SparkMPI/data/gddp/tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2010.tif', 17)
pathList = [ (os.path.join(basepath, fileNames[i]), j) for i in range(len(fileNames)) for j in range(365) ]

data = sc.parallelize(pathList, 24)

rdd = data.map(readDay)                    \
    .reduceByKey(sumDays)                  \
    .map(average)                          \
    #.sortByKey()                           \
    #.coalesce(nbMPIPartition)              \
    #.mapPartitionsWithIndex(visualization) \
    .collect()

# t1 = time.time()
# print('### Total execution time - %s | ' % str(t1 - t0))

# print('### Stop execution - %s' % str(datetime.now()))
