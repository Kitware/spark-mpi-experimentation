from __future__ import print_function

import os
import sys
import pyspark

from pyspark import SparkContext

from paraview import simple
import vtk
import numpy as np
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support

sc = SparkContext()

# Define the address of the PMI server and the number of MPI workers

hostname = os.uname()[1]
hydra_proxy_port = os.getenv("HYDRA_PROXY_PORT")
pmi_port = hostname + ":" + hydra_proxy_port

sc.getConf().set('mpi', pmi_port)

# -------------------------------------------------------------------------
# Read Tiff file
# -------------------------------------------------------------------------

filePath = '/data/sebastien/SparkMPI/data-convert/etl/Recon_NanoParticle_doi_10.1021-nl103400a.tif'
reader = simple.TIFFSeriesReader(FileNames=[filePath])
reader.UpdatePipeline()
imageData = dsa.WrapDataObject(reader.GetClientSideObject().GetOutputDataObject(0))
originalDimensions = imageData.VTKObject.GetDimensions()
arrayNames = imageData.PointData.keys()
print('arrayNames', arrayNames)
dataArray = imageData.PointData[arrayNames[0]]
#print(dataArray)
#npArray = numpy_support.vtk_to_numpy(dataArray)
#print(npArray)

# -------------------------------------------------------------------------

targetPartition = 4

sizeX = originalDimensions[0]
sizeY = originalDimensions[1]
sizeZ = originalDimensions[2]
sliceSize = sizeX * sizeY

globalMaxIndex = sizeX * sizeY * sizeZ

deltaZ = float(sizeZ) / float(targetPartition)
zSizes = [(int((i+1) * deltaZ) - int(i * deltaZ)) for i in range(targetPartition)]


def getPartition(value):
    count = 0
    threshold = 0
    for z in zSizes:
        threshold += z * sliceSize
        if value < threshold:
            return count
        count += 1
    return count

# -------------------------------------------------------------------------

rdd = sc.parallelize(dataArray).zipWithIndex()
rdd2 = rdd.partitionBy(targetPartition, getPartition)

print('number of partitions', rdd2.getNumPartitions())

# -------------------------------------------------------------------------

def idxToCoord(idx):
    return [
        (idx % sizeX),
        (idx / sizeX % sizeY),
        int(idx / sliceSize),
    ]

def processPartition(idx, iterator):
    print('==============> processPartition %d' % idx)
    import os
    os.environ["PMI_PORT"] = pmi_port
    os.environ["PMI_ID"] = str(idx)
    os.environ["PV_ALLOW_BATCH_INTERACTION"] = "1"
    os.environ["DISPLAY"] = ":0"

    import paraview
    paraview.options.batch = True

    from vtk.vtkCommonExecutionModel import vtkTrivialProducer
    from vtk.vtkPVVTKExtensionsCore import vtkDistributedTrivialProducer
    from vtk.vtkCommonCore import vtkIntArray, vtkUnsignedCharArray
    from vtk.vtkCommonDataModel import vtkImageData, vtkPointData, vtkCellData
        
    offset = 0
    for i in range(idx):
        offset += zSizes[i] * sliceSize
    size = zSizes[idx] * sliceSize

    maxIdx = 0
    minIdx = globalMaxIndex
    
    data = vtkIntArray()
    data.SetName('scalar')
    data.SetNumberOfTuples(size)
    data.Fill(0)

    mask = vtkUnsignedCharArray()
    mask.SetName('vtkGhostType')
    mask.SetNumberOfTuples(size)
    mask.Fill(32) # HIDDENCELL

    for row in iterator:
        #mask.SetTuple1(row[0] - offset, 1) # Duplicate
        data.SetValue(row[0] - offset, row[1])
        maxIdx = row[0] if row[0] > maxIdx else maxIdx
        minIdx = row[0] if row[0] < minIdx else minIdx

    # FIXME should shrink array now

    print('%d - (%d, %d) - offset %d - size %d - dimension %d, %d, %d ' % (idx, minIdx - offset, maxIdx - offset, offset, size, sizeX, sizeY, zSizes[idx]))

    dataset = vtkImageData()
    # dataset.SetDimensions(sizeX, sizeY, zSizes[idx])
    minZ = 0
    maxZ = 0
    for i in range(idx + 1):
        minZ = maxZ
        maxZ += zSizes[i]

    print('%d extent %d, %d' % (idx, minZ, maxZ - 1))
    dataset.SetExtent(0, sizeX - 1, 0, sizeY - 1, minZ, maxZ - 1)
    # dataset.SetOrigin(0, 0, offset / sliceSize - idx)
    dataset.GetPointData().SetScalars(data)
    #dataset.GetCellData().AddArray(mask)
    #dataset.GetCellData().AddArray(pid)

    vtkDistributedTrivialProducer.SetGlobalOutput('Spark', dataset)

    from vtk.vtkPVClientServerCoreCore import vtkProcessModule
    from paraview     import simple
    from paraview.web import wamp      as pv_wamp
    from paraview.web import protocols as pv_protocols
    from vtk.web      import server

    pm = vtkProcessModule.GetProcessModule()

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

    class _VisualizerServer(pv_wamp.PVServerProtocol):
        dataDir = '/data'
        groupRegex = "[0-9]+\\.[0-9]+\\.|[0-9]+\\."
        excludeRegex = "^\\.|~$|^\\$"
        allReaders = True
        viewportScale=1.0
        viewportMaxWidth=2560
        viewportMaxHeight=1440
        proxies='/data/sebastien/SparkMPI/defaultProxies.json'

        def initialize(self):
            # Bring used components
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebFileListing(_VisualizerServer.dataDir, "Home", _VisualizerServer.excludeRegex, _VisualizerServer.groupRegex))
            self.registerVtkWebProtocol(pv_protocols.ParaViewWebProxyManager(baseDir=_VisualizerServer.dataDir, allowedProxiesFile=_VisualizerServer.proxies, allowUnconfiguredReaders=_VisualizerServer.allReaders))
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

    args = Options()
    if pm.GetPartitionId() == 0:
        producer = simple.DistributedTrivialProducer()
        producer.UpdateDataset = ''
        producer.UpdateDataset = 'Spark'
        producer.WholeExtent = [0, 99, 0, 215, 0, 260]
        server.start_webserver(options=args, protocol=_VisualizerServer)
        pm.GetGlobalController().TriggerBreakRMIs()
            
    yield (idx, targetPartition)

# -------------------------------------------------------------------------

results = rdd2.mapPartitionsWithIndex(processPartition).collect()
for out in results:
    print(out)

