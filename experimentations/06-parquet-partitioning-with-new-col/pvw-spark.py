from __future__ import print_function

import os
import sys
import pyspark

from pyspark import SparkContext
from pyspark.sql import SparkSession

from pyspark.sql.functions import udf
from pyspark.sql.types import *

sc = SparkContext()
spark = SparkSession.builder.appName("Python Spark SQL basic example").getOrCreate()

# Define the address of the PMI server and the number of MPI workers

hostname = os.uname()[1]
hydra_proxy_port = os.getenv("HYDRA_PROXY_PORT")
pmi_port = hostname + ":" + hydra_proxy_port

sc.getConf().set('mpi', pmi_port)
maxIndex = 100 * 216 * 261

# -------------------------------------------------------------------------

sqc = pyspark.sql.SQLContext(sc)
df = sqc.read.parquet("/data/sebastien/SparkMPI/data/data-xyzv.parq")

maxZ = df.agg({"z": "max"}).collect()[0]['max(z)']

print("maxZ", maxZ)

# Add column for partition
def zToPartition(value):
    return int(4 * 4 * value / (maxZ + 1))

udfOp = udf(zToPartition, IntegerType())
dfp = df.withColumn("partition", udfOp("z"))

rdd = dfp.repartition(4, 'partition').rdd # FIXME

partition = rdd.getNumPartitions()
print('number of partitions', partition)

# -------------------------------------------------------------------------

def processPartition(idx, iterator):
    import os
    os.environ["PMI_PORT"] = pmi_port
    os.environ["PMI_ID"] = str(idx)
    os.environ["PV_ALLOW_BATCH_INTERACTION"] = "1"
    os.environ["DISPLAY"] = ":0"

    import paraview
    paraview.options.batch = True

    from vtk.vtkPVVTKExtensionsCore import vtkDistributedTrivialProducer
    from vtk.vtkCommonCore import vtkIntArray, vtkPoints
    from vtk.vtkCommonDataModel import vtkPolyData, vtkPointData, vtkCellArray

    pointData = vtkIntArray()
    pointData.SetName('scalar')
    pointData.Allocate(maxIndex)

    points = vtkPoints()
    points.Allocate(maxIndex)

    for row in iterator:
        points.InsertNextPoint(row['x'], row['y'], row['z'])
        pointData.InsertNextTuple1(row['value'])

    cells = vtkCellArray()
    cells.Allocate(points.GetNumberOfPoints() + 1)
    cells.InsertNextCell(points.GetNumberOfPoints())
    for i in range(points.GetNumberOfPoints()):
        cells.InsertCellPoint(i)

    dataset = vtkPolyData()
    dataset.SetPoints(points)
    dataset.SetVerts(cells)
    dataset.GetPointData().SetScalars(pointData)

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

    args = Options()
    if pm.GetPartitionId() == 0:
        producer = simple.DistributedTrivialProducer()
        producer.UpdateDataset = ''
        producer.UpdateDataset = 'Spark'
        server.start_webserver(options=args, protocol=_VisualizerServer)
        pm.GetGlobalController().TriggerBreakRMIs()

    yield (idx, partition)

# -------------------------------------------------------------------------

results = rdd.mapPartitionsWithIndex(processPartition).collect()
for out in results:
    print(out)

