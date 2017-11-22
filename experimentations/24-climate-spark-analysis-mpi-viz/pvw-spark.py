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
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2009.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2010.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2011.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2012.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2013.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2014.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2015.tif',
]
allYearsList = [ f[-8:-4] for f in fileNames ]
basepath = '/data/scott/SparkMPI/data/gddp'

# -------------------------------------------------------------------------
# Module-level variables and helper functions
# -------------------------------------------------------------------------

sizeZ = len(fileNames)

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

nbMPIPartition = int(os.environ["MPI_SIZE"])
npSparkPartition = int(os.environ["SPARK_SIZE"])

# -------------------------------------------------------------------------
# Partition handling
# -------------------------------------------------------------------------

sc = SparkContext()

# -------------------------------------------------------------------------
# MPI configuration
# -------------------------------------------------------------------------

hostname = os.uname()[1]
hydra_proxy_port = os.getenv("HYDRA_PROXY_PORT")
pmi_port = hostname + ":" + hydra_proxy_port

# -----------------------------------------------------------------------------
# ParaViewWeb Options
# -----------------------------------------------------------------------------

class Options(object):
    debug = False
    nosignalhandlers = True
    host = 'localhost'
    port = 9753
    timeout = 300
    content = '/data/scott/SparkMPI/runtime/visualizer-2.1.4/dist'
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
# Visualization
# -------------------------------------------------------------------------

def visualization(partitionId, iterator):
    # Setup MPI context
    import os
    os.environ["PMI_PORT"] = pmi_port
    os.environ["PMI_ID"] = str(partitionId)
    os.environ["PV_ALLOW_BATCH_INTERACTION"] = "1"
    os.environ["DISPLAY"] = ":0"

    numYears = 0
    localYears = []
    localAvgData = []
    for item in iterator:
        numYears += 1
        print('visualize (partition %d) peeking at next year %s, shape = [%d, %d]' % (partitionId, item[0], item[1].shape[1], item[1].shape[0]))
        # print(item[1])
        localYears.append(item[0])
        localAvgData.append(item[1])

    sizeX = localAvgData[0].shape[1]
    sizeY = localAvgData[0].shape[0]

    sliceSize = sizeX * sizeY
    localNumSlices = numYears
    size = localNumSlices * sliceSize

    print('visualize, partition id = %d, sizeX = %d, sizeY = %d, sliceSize = %d, localNumSlices = %d, size = %d' % (partitionId, sizeX, sizeY, sliceSize, localNumSlices, size))

    # # Copy data from iterator into data chunk
    t0 = time.time()
    count = 0
    globalSliceIndices = []
    dataChunk = np.arange(size, dtype=np.float32)
    sidChunk = np.arange(size, dtype=np.uint8)
    for localSliceIdx in range(len(localYears)):
        globalSliceIdx = allYearsList.index(localYears[localSliceIdx])
        globalSliceIndices.append(globalSliceIdx)
        nextAvgArray = localAvgData[localSliceIdx]
        (imgHeight, imgWidth) = nextAvgArray.shape
        for y in range(imgHeight):
            for x in range(imgWidth):
                count += 1
                pixelValue = nextAvgArray[y][x]
                destIdx = x + (y * sizeX) + (localSliceIdx * sliceSize)
                dataChunk[destIdx] = pixelValue
                sidChunk[destIdx] = globalSliceIdx

    t1 = time.time()
    print('%d # MPI Gather %s | %d' % (partitionId, str(t1 - t0), count))
    t0 = t1

    # Configure Paraview for MPI
    import paraview
    paraview.options.batch = True

    from vtk.vtkPVVTKExtensionsCore import vtkDistributedTrivialProducer
    from vtk.vtkCommonCore import vtkIntArray, vtkUnsignedCharArray, vtkFloatArray
    from vtk.vtkCommonDataModel import vtkImageData, vtkPointData

    # # -------------------------------------------------------------------------

    dataset = vtkImageData()

    sliceIdxArray = vtkUnsignedCharArray()
    sliceIdxArray.SetName('Slice Index')
    sliceIdxArray.SetNumberOfComponents(1)
    sliceIdxArray.SetNumberOfTuples(size)

    dataArray = vtkFloatArray()
    dataArray.SetName('Annual Avg Temp')
    dataArray.SetNumberOfComponents(1)
    dataArray.SetNumberOfTuples(size)
    for i in range(size):
        dataArray.SetTuple1(i, dataChunk[i])
        sliceIdxArray.SetTuple1(i, sidChunk[i])

    minZ = globalSliceIndices[0]
    maxZ = globalSliceIndices[-1]

    # dataset.SetExtent(0, sizeX - 1, 0, sizeY - 1, minZ, maxZ - 1)
    # print('partition %d extents: [%d, %d, %d, %d, %d, %d]' % (partitionId, 0, sizeX - 1, 0, sizeY - 1, minZ, maxZ - 1))
    dataset.SetExtent(0, sizeX - 1, 0, sizeY - 1, minZ, maxZ)
    print('partition %d extents: [%d, %d, %d, %d, %d, %d]' % (partitionId, 0, sizeX - 1, 0, sizeY - 1, minZ, maxZ))
    #dataset.GetPointData().AddArray(dataArray)
    # dataset.GetPointData().SetScalars(dataArray)
    dataset.GetCellData().SetScalars(dataArray)

    procIdArray = vtkUnsignedCharArray()
    procIdArray.SetName('Process Id')
    procIdArray.SetNumberOfComponents(1)
    procIdArray.SetNumberOfTuples(size)

    for i in range(size):
        procIdArray.SetTuple1(i, partitionId)

    dataset.GetCellData().AddArray(procIdArray)
    dataset.GetCellData().AddArray(sliceIdxArray)

    t1 = time.time()
    print('%d # build resulting image data %s | ' % (partitionId, str(t1 - t0)))
    t0 = t1

    # -------------------------------------------------------------------------

    print('%d about to set global output on producer' % partitionId)
    print(dataset)
    vtkDistributedTrivialProducer.SetGlobalOutput('Spark', dataset)
    print('%d just set global output on producer' % partitionId)

    from vtk.vtkPVClientServerCoreCore import vtkProcessModule
    from paraview     import simple
    from vtk.web      import server
    from paraview.web import wamp as pv_wamp
    from paraview.web import protocols as pv_protocols

    class _VisualizerServer(pv_wamp.PVServerProtocol):
        dataDir = '/data'
        groupRegex = "[0-9]+\\.[0-9]+\\.|[0-9]+\\."
        excludeRegex = "^\\.|~$|^\\$"
        allReaders = True
        viewportScale=1.0
        viewportMaxWidth=2560
        viewportMaxHeight=1440

        def initialize(self):
            # Bring used components
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebFileListing(_VisualizerServer.dataDir, "Home", _VisualizerServer.excludeRegex, _VisualizerServer.groupRegex))
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebProxyManager(baseDir=_VisualizerServer.dataDir, allowUnconfiguredReaders=_VisualizerServer.allReaders))
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebColorManager())
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebMouseHandler())
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebViewPort(_VisualizerServer.viewportScale, _VisualizerServer.viewportMaxWidth,
                                                                         _VisualizerServer.viewportMaxHeight))
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebViewPortImageDelivery())
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebViewPortGeometryDelivery())
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebTimeHandler())
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebSelectionHandler())
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebWidgetManager())
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebKeyValuePairStore())
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebSaveData(baseSavePath=_VisualizerServer.dataDir))
            # Disable interactor-based render calls
            simple.GetRenderView().EnableRenderOnInteraction = 0
            simple.GetRenderView().Background = [0,0,0]
            # Update interaction mode
            pxm = simple.servermanager.ProxyManager()
            interactionProxy = pxm.GetProxy('settings', 'RenderViewInteractionSettings')
            interactionProxy.Camera3DManipulators = ['Rotate', 'Pan', 'Zoom', 'Pan', 'Roll', 'Pan', 'Zoom', 'Rotate', 'Zoom']


    print('%d about to GetProcessModule' % partitionId)
    pm = vtkProcessModule.GetProcessModule()
    print('%d successfully got process module' % partitionId)

    # -------------------------------------------------------------------------

    print('%d # > Start visualization - %s | ' % (partitionId, str(datetime.now())))

    # -------------------------------------------------------------------------

    args = Options()
    if pm.GetPartitionId() == 0:
        print('%d # ==> %d' % (partitionId, pm.GetPartitionId()))
        producer = simple.DistributedTrivialProducer()
        producer.UpdateDataset = ''
        producer.UpdateDataset = 'Spark'
        print('rank 0 process setting whole extent to [%d, %d, %d, %d, %d, %d]' % (0, sizeX - 1, 0, sizeY - 1, 0, sizeZ - 1))
        producer.WholeExtent = [0, sizeX - 1, 0, sizeY - 1, 0, sizeZ - 1]
        server.start_webserver(options=args, protocol=_VisualizerServer)
        pm.GetGlobalController().TriggerBreakRMIs()

    print('%d # < Stop visualization - %s | ' % (partitionId, str(datetime.now())))
    yield (partitionId, nbMPIPartition)

# -------------------------------------------------------------------------
# Spark pipeline
# -------------------------------------------------------------------------

print('Starting processing, # spark partitions = %d, # mpi partitions = %d' % (npSparkPartition, nbMPIPartition))

# pathList will contain 3650 entries like ('/data/scott/SparkMPI/data/gddp/tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2010.tif', 17)
pathList = [ (os.path.join(basepath, fileNames[i]), j) for i in range(len(fileNames)) for j in range(365) ]

data = sc.parallelize(pathList, npSparkPartition)

rdd = data.map(readDay)                    \
    .reduceByKey(sumDays, numPartitions=len(fileNames)) \
    .map(average)                          \
    .sortByKey()                           \
    .coalesce(nbMPIPartition)              \
    .mapPartitionsWithIndex(visualization) \
    .collect()

    # .glom() \
    # .collect()

print(rdd)

#
#
# .mapPartitionsWithIndex(visualization) \

# t1 = time.time()
# print('### Total execution time - %s | ' % str(t1 - t0))

# print('### Stop execution - %s' % str(datetime.now()))
