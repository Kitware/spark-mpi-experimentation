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

def processFile(fileName):
    # junk = raw_input('blah')

    print('Processing %s' % fileName)

    year = fileName.split('_')[-1][:-4]
    dataset = gdal.Open(fileName)
    numBands = dataset.RasterCount

    imgDims = [dataset.RasterXSize, dataset.RasterYSize]

    imgData = vtk.vtkImageData()
    imgData.SetDimensions(imgDims[0], imgDims[1], 0)
    imgData.SetSpacing(1, 1, 1)
    imgData.SetOrigin(0, 0, 0)
    imgData.SetExtent(0, imgDims[0] - 1, 0, imgDims[1] - 1, 0, 0)

    pointData = imgData.GetPointData()

    # for bandId in range(numBands):
    for bandId in [0, 89, 179, 269]:
        print('  band %d' % (bandId + 1))
        band = dataset.GetRasterBand(bandId + 1).ReadAsArray()

        dataArray = numpy_support.numpy_to_vtk(np.ndarray.flatten(band[::-1,:], 0).astype(np.dtype(np.int32)), deep = 1)
        dataArray.SetName('Daily Max (%d)' % (bandId + 1))
        pointData.AddArray(dataArray)

    pointData.SetActiveScalars('Daily Max (0)')

    imgWriter = vtkIOXML.vtkXMLImageDataWriter()
    imgWriter.SetFileName('tasmax_%s.vti' % (year))
    imgWriter.SetInputData(imgData)
    imgWriter.Write()

# ---------------------------------------------------------------------------

# for fileName in fileNames:
#     fpath = os.path.join(basepath, fileName)
#     processFile(fpath)

processFile(os.path.join(basepath, fileNames[0]))
