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


def computeAveragesAndBuildImgData():
    imgDims = [1440, 720, 10]

    imgData = vtk.vtkImageData()
    imgData.SetDimensions(imgDims[0] + 1, imgDims[1] + 1, imgDims[2] + 1)
    imgData.SetSpacing(1, 1, 1)
    imgData.SetOrigin(0, 0, 0)
    imgData.SetExtent(0, imgDims[0], 0, imgDims[1], 0, imgDims[2])
    imgData.AllocateScalars(vtk.VTK_FLOAT, 1)

    flattenedArrays = []

    for fileName in fileNames:
        fpath = os.path.join(basepath, fileName)
        print('processing %s' % fpath)

        year = fileName.split('_')[-1][:-4]

        dataset = gdal.Open(fpath)
    
        sumArray = ma.zeros((dataset.RasterYSize, dataset.RasterXSize))
        total = 0
        count = 0
        numBands = dataset.RasterCount
    
        for bandId in range(numBands):
            band = ma.masked_outside(dataset.GetRasterBand(bandId + 1).ReadAsArray(), VALUE_RANGE[0], VALUE_RANGE[1])
            sumArray += band
    
        sumArray /= numBands
        total = ma.sum(ma.sum(sumArray))
        count = sumArray.count()
        minCell = ma.min(sumArray)
        maxCell = ma.max(sumArray)
        imgDims = [dataset.RasterXSize, dataset.RasterYSize]

        # flattenedArrays.append(np.ndarray.flatten(sumArray[::-1,:], 0).astype(np.dtype(np.float32)))
        flattenedArrays.append(np.ndarray.flatten(sumArray[::-1,:], 0))

    allYearsAvgs = np.ma.concatenate(flattenedArrays)

    cellData = imgData.GetCellData()
    dataArray = numpy_support.numpy_to_vtk(np.ndarray.flatten(allYearsAvgs, 0), deep = 1)
    dataArray.SetName('Annual Avg Temp')
    cellData.SetScalars(dataArray)

    imgWriter = vtkIOXML.vtkXMLImageDataWriter()
    imgWriter.SetFileName('/data/scott/Documents/tasmax.vti')
    imgWriter.SetInputData(imgData)
    imgWriter.Write()

    print('\nFinished writing image data\n')


computeAveragesAndBuildImgData()
