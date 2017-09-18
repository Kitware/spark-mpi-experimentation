from __future__ import print_function
import os
import sys
import time
import gdal
from datetime import datetime

from paraview import vtk
from paraview.vtk import vtkIOXML
from vtk.util import numpy_support

import numpy as np
import numpy.ma as ma

VALUE_RANGE = [0, 50000]

def kToF(degKelvin):
    return ((degKelvin - 273.15) * (9 / 5.0)) + 32

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

# ---------------------------------------------------------------------------

def readFileAndCompute(fileName):
    year = fileName.split('_')[-1][:-4]

    dataset = gdal.Open(fileName)

    # print('dataset methods and attributes:')
    # print(dir(dataset))
    # print()

    # metadata = dataset.GetMetadata()

    # print('dataset metadata:')
    # for k in metadata:
       #  print('%s => %s' % (k, metadata[k]))
    # print()
    # print('  raster size: x => %d, y => %d' % (dataset.RasterXSize, dataset.RasterYSize))

    sumArray = ma.zeros((dataset.RasterYSize, dataset.RasterXSize))
    total = 0
    count = 0
    numBands = dataset.RasterCount

    # print('  number of bands: %d' % numBands)
    # numBands = 10

    for bandId in range(numBands):
        # print('processing band %d' % (bandId + 1))
        band = ma.masked_outside(dataset.GetRasterBand(bandId + 1).ReadAsArray(), VALUE_RANGE[0], VALUE_RANGE[1])
        sumArray += band

    sumArray /= numBands
    total = ma.sum(ma.sum(sumArray))
    count = sumArray.count()
    minCell = ma.min(sumArray)
    maxCell = ma.max(sumArray)

    # print('  final sum array has %d non-masked values' % count)

    return (year, [dataset.RasterXSize, dataset.RasterYSize], minCell, maxCell, total / count, sumArray)

# ---------------------------------------------------------------------------

def processFile(fileName):
    print('Processing %s' % fileName)
    year, imgDims, mint, maxt, avgt, avgBand = readFileAndCompute(fileName)

    print('    Year: %s, min avg temp: %f deg F, max avg temp: %f deg F, earth avg temp: %f deg F' % (year, kToF(mint), kToF(maxt), kToF(avgt)))

    imgData = vtk.vtkImageData()
    imgData.SetDimensions(imgDims[0], imgDims[1], 0)
    imgData.SetSpacing(1, 1, 1)
    imgData.SetOrigin(0, 0, 0)
    imgData.SetExtent(0, imgDims[0] - 1, 0, imgDims[1] - 1, 0, 0)
    imgData.AllocateScalars(vtk.VTK_FLOAT, 1)

    pointData = imgData.GetPointData()
    dataArray = numpy_support.numpy_to_vtk(np.ndarray.flatten(avgBand[::-1,:], 0))
    dataArray.SetName('Annual Avg Temp')
    pointData.SetScalars(dataArray)

    imgWriter = vtkIOXML.vtkXMLImageDataWriter()
    imgWriter.SetFileName('tasmax_%s.vti' % (year))
    imgWriter.SetInputData(imgData)
    imgWriter.Write()

# ---------------------------------------------------------------------------

# for fileName in fileNames:
#     fpath = os.path.join(basepath, fileName)
#     processFile(fpath)

processFile(os.path.join(basepath, fileNames[0]))
