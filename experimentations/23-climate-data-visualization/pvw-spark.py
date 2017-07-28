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
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2008.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2009.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2010.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2011.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2012.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2013.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2014.tif',
    # 'tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2015.tif',
]
basepath = '/data/scott/SparkMPI/data/gddp'

# -------------------------------------------------------------------------
# Module-level variables and helper functions
# -------------------------------------------------------------------------

sizeX = 0
sizeY = 0
sizeZ = 0
sliceSize = 0
globalMaxIndex = 0

mpiStepX = 0
mpiStepX_ = 0

processSlices = []

def _ijk(index):
    return [
        (index % sizeX),
        (index / sizeX % sizeY),
        int(index / sliceSize),
    ]

def swap(kv):
    return (kv[1], kv[0])

# -------------------------------------------------------------------------
# Function to compute the average image for all years
# -------------------------------------------------------------------------

def computeAveragesUsingNumpy():
    global sizeX, sizeY, sizeZ
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
        sizeX = dataset.RasterXSize
        sizeY = dataset.RasterYSize

        flattenedArrays.append(np.ndarray.flatten(sumArray[::-1,:], 0).astype(np.dtype(np.int32)))

    sizeZ = len(flattenedArrays)

    return np.ma.concatenate(flattenedArrays)

# def computeAveragesUsingNumpy():
#     global sizeX, sizeY, sizeZ
#     sizeX = 1440
#     sizeY = 720
#     sizeZ = 10

# -------------------------------------------------------------------------
# Parallel configuration
# -------------------------------------------------------------------------

nbMPIPartition = int(os.environ["MPI_SIZE"])

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

    localNumSlices = processSlices[partitionId]
    size = localNumSlices * sliceSize
    kOffset = 0
    for i in range(partitionId):
        kOffset += processSlices[i]

    # Copy data from iterator into data chunk
    t0 = time.time()
    count = 0
    dataChunk = np.arange(size, dtype=np.int32)
    for item in iterator:
        count += 1
        globalIndex = item[0]
        pixelValue = item[1]
        # print('%d # %d: %f' % (partitionId, globalIndex, pixelValue))
        ijk = _ijk(globalIndex)
        ijk[2] -= kOffset
        destIdx = ijk[0] + (ijk[1] * sizeX)  + (ijk[2] * sliceSize)
        dataChunk[destIdx] = pixelValue

    t1 = time.time()
    print('%d # MPI Gather %s | %d' % (partitionId, str(t1 - t0), count))
    t0 = t1

    # Configure Paraview for MPI
    import paraview
    paraview.options.batch = True

    from vtk.vtkPVVTKExtensionsCore import vtkDistributedTrivialProducer
    from vtk.vtkCommonCore import vtkIntArray, vtkUnsignedCharArray, vtkFloatArray
    from vtk.vtkCommonDataModel import vtkImageData, vtkPointData

    # -------------------------------------------------------------------------

    dataset = vtkImageData()
    dataArray = vtkIntArray()
    dataArray.SetName('Annual Avg Temp')
    dataArray.SetNumberOfComponents(1)
    dataArray.SetNumberOfTuples(size)
    for i in range(size):
        dataArray.SetTuple1(i, dataChunk[i])

    minZ = 0
    maxZ = 0
    for i in range(partitionId + 1):
        minZ = maxZ
        maxZ += processSlices[i]

    dataset.SetExtent(0, sizeX - 1, 0, sizeY - 1, minZ, maxZ - 1)
    dataset.GetPointData().SetScalars(dataArray)

    t1 = time.time()
    print('%d # build resulting image data %s | ' % (partitionId, str(t1 - t0)))
    t0 = t1

    # -------------------------------------------------------------------------

    vtkDistributedTrivialProducer.SetGlobalOutput('Spark', dataset)

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


    pm = vtkProcessModule.GetProcessModule()

    # -------------------------------------------------------------------------

    print('%d # > Start visualization - %s | ' % (partitionId, str(datetime.now())))

    # -------------------------------------------------------------------------

    args = Options()
    if pm.GetPartitionId() == 0:
        print('%d # ==> %d' % (partitionId, pm.GetPartitionId()))
        producer = simple.DistributedTrivialProducer()
        producer.UpdateDataset = ''
        producer.UpdateDataset = 'Spark'
        producer.WholeExtent = [0, sizeX - 1, 0, sizeY - 1, 0, sizeY - 1]
        server.start_webserver(options=args, protocol=_VisualizerServer)
        pm.GetGlobalController().TriggerBreakRMIs()

    print('%d # < Stop visualization - %s | ' % (partitionId, str(datetime.now())))
    yield (partitionId, nbMPIPartition)

# -------------------------------------------------------------------------
# Spark pipeline
# -------------------------------------------------------------------------

npArray = computeAveragesUsingNumpy()

sliceSize = sizeX * sizeY
print('slice size: %d' % sliceSize)

globalMaxIndex = sizeX * sizeY * sizeZ

processSlices = [ sizeZ / nbMPIPartition for i in range(nbMPIPartition) ]
for i in range(sizeZ % nbMPIPartition):
    processSlices[i] += 1

print('Dimensions: %d x %d x %d' % (sizeX, sizeY, sizeZ))
print('number of slices per partition: ', processSlices)

# partitions = [ [] for i in range(nbMPIPartition)]
# for i in range(globalMaxIndex):
#     partIdx = getMPIPartition(i)
#     partitions[partIdx].append(i)

# print('division of work: ')
# for i in range(len(partitions)):
#     print('  process %d will handle %d pixels' % (i, len(partitions[i])))

t0 = time.time()
data = sc.parallelize(npArray)

rdd = data.zipWithIndex() \
    .map(swap) \
    .partitionBy(nbMPIPartition, getMPIPartition) \
    .mapPartitionsWithIndex(visualization) \
    .collect()

t1 = time.time()
print('### Total execution time - %s | ' % str(t1 - t0))

print('### Stop execution - %s' % str(datetime.now()))
