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
import numpy as np
import scipy.sparse as ss
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support
from vtk.vtkCommonCore import vtkIntArray


sc = SparkContext()

# -------------------------------------------------------------------------
# MPI configuration
# -------------------------------------------------------------------------

hostname = os.uname()[1]
hydra_proxy_port = os.getenv("HYDRA_PROXY_PORT")
pmi_port = hostname + ":" + hydra_proxy_port

# -------------------------------------------------------------------------
# Parallel configuration
# -------------------------------------------------------------------------

nbSparkPartition = int(os.environ["SPARK_SIZE"])
nbMPIPartition = int(os.environ["MPI_SIZE"])

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
# Remove invalid value
# -----------------------------------------------------------------------------

def valid(value):
    return value[1] < 5000

# -----------------------------------------------------------------------------
# Sum
# -----------------------------------------------------------------------------

def sum(v1, v2):
    return (v1[0] + v2[0], v1[1] + v2[1])

# -----------------------------------------------------------------------------
# Count groups
# -----------------------------------------------------------------------------

def addCount(kv):
    return (kv[0], (kv[1], 1))

# -----------------------------------------------------------------------------
# Avg
# -----------------------------------------------------------------------------

def avg(kv):
    return (kv[0], kv[1][0] / kv[1][1])

# -----------------------------------------------------------------------------
# ParaViewWeb Options
# -----------------------------------------------------------------------------

class Options(object):
    debug = False
    nosignalhandlers = True
    host = 'localhost'
    port = 9753
    timeout = 300
    content = '/data/sebastien/SparkMPI/runtime/visualizer/dist'
    forceFlush = False
    sslKey = ''
    sslCert = ''
    ws = 'ws'
    lp = 'lp'
    hp = 'hp'
    nows = False
    nobws = False
    nolp = False
    fsEndpoints = ''
    uploadPath = None
    testScriptPath = ''
    baselineImgDir = ''
    useBrowser = 'nobrowser'
    tmpDirectory = '.'
    testImgFile = ''

# -------------------------------------------------------------------------
# Spark pipeline 1
# -------------------------------------------------------------------------

# t0 = time.time()
# data = sc.parallelize(fileNames) \
#          .repartition(len(fileNames)) \
#          .flatMap(readFile) \
#          .repartition(nbSparkPartition) \
#          .filter(valid) \
#          .map(addCount) \
#          .reduceByKey(sum) \
#          .map(avg) \
#          .collect()
# t1 = time.time()
# print('### Total execution time - %s ' % str(t1 - t0))
#
# for r in data:
#     print(r)
#
# print('### Stop execution - %s' % str(datetime.now()))


# -------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------

# ### Total execution time - 5277.02247286
# ('2006', 22.890404789604645)
# ('2013', 22.889763551578667)
# ('2007', 22.889763551578667)
# ('2012', 22.827223463966494)
# ('2015', 22.889763551578667)
# ('2014', 22.889763551578667)
# ('2008', 22.827223463966494)
# ('2009', 22.889763551578667)
# ('2011', 22.890426618984254)
# ('2010', 22.889763551578667)
# ### Stop execution - 2017-06-21 15:14:22.528466

# -------------------------------------------------------------------------
# Spark pipeline 2
# -------------------------------------------------------------------------

t0 = time.time()
data = sc.parallelize(fileNames) \
         .repartition(len(fileNames)) \
         .map(readFileAndCompute) \
         .collect()
t1 = time.time()
print('### Total execution time - %s ' % str(t1 - t0))

for r in data:
    print(r)

print('### Stop execution - %s' % str(datetime.now()))


readFileAndCompute
