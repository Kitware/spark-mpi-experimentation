from __future__ import print_function
import os
import sys
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

# Define the address of the PMI server and the number of MPI workers

hostname = os.uname()[1]
hydra_proxy_port = os.getenv("HYDRA_PROXY_PORT")
pmi_port = hostname + ":" + hydra_proxy_port

sc.getConf().set('mpi', pmi_port)

# -------------------------------------------------------------------------
# Read Tiff file
# -------------------------------------------------------------------------

filePath = '/data/sebastien/SparkMPI/data-convert/etl/TiltSeries_NanoParticle_doi_10.1021-nl103400a.tif'
reader = simple.TIFFSeriesReader(FileNames=[filePath])
reader.UpdatePipeline()
imageData = reader.GetClientSideObject().GetOutputDataObject(0)
originalDimensions = imageData.GetDimensions()
dataArray = imageData.GetPointData().GetScalars()
intArray = vtkIntArray()
intArray.SetNumberOfTuples(dataArray.GetNumberOfTuples())
for idx in range(dataArray.GetNumberOfTuples()):
    intArray.SetTuple1(idx, int(dataArray.GetTuple1(idx)))

print('range', dataArray.GetRange(), intArray.GetRange())
print('GetNumberOfTuples', dataArray.GetNumberOfTuples(), intArray.GetNumberOfTuples())
npArray = numpy_support.vtk_to_numpy(intArray)

# -------------------------------------------------------------------------

targetPartition = 4

sizeX = originalDimensions[0]
sizeY = originalDimensions[1]
sizeZ = originalDimensions[2]
sliceSize = sizeX * sizeY

print('Dimensions: %d x %d x %d' % (sizeX, sizeY, sizeZ))
globalMaxIndex = sizeX * sizeY * sizeZ

deltaX = int(float(sizeX) / float(targetPartition))
xSizes = []
for i in range(targetPartition):
    xSizes.append(deltaX)
xSizes[-1] += sizeX % targetPartition

def getPartition(value):
    i = value % sizeX
    return int(targetPartition * i / sizeX)

# -------------------------------------------------------------------------

data = sc.parallelize(npArray)
index = sc.parallelize(range(globalMaxIndex))
rdd = index.zip(data)
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

    iOffset = 0
    for i in range(idx):
        iOffset += xSizes[i]
    size = xSizes[idx] * sizeY * sizeZ

    # -------------------------------------------------------------------------
    # Gather data chunk
    # -------------------------------------------------------------------------

    dataChunk = vtkIntArray()
    dataChunk.SetName('scalar')
    dataChunk.SetNumberOfTuples(size)
    dataChunk.Fill(0)

    print('%d # Reserve chunk size (%d, %d, %d) - size: %d ' % (idx, xSizes[idx], sizeY, sizeZ, size))

    count = 0
    for row in iterator:
        count += 1
        coords = idxToCoord(row[0])
        originCoords = '[%d, %d, %d]' % (coords[0], coords[1], coords[2])
        coords[0] -= iOffset
        destIdx = coords[0] + (coords[1] * xSizes[idx])  + (coords[2] * xSizes[idx] * sizeY)
        if destIdx < 0 or destIdx > size:
            print('(%d) %d => %s | (%d, %d, %d) => %d | [%d, %d, %d]' % (idx, row[0], originCoords, coords[0], coords[1], coords[2], destIdx, xSizes[idx], sizeY, sizeZ))
        else:
            dataChunk.SetValue(destIdx, row[1])

    print('%d # Data chunk filled (%d, %d, %d) - size: %d - filled: %d' % (idx, xSizes[idx], sizeY, sizeZ, size, count))

    # -------------------------------------------------------------------------
    # Reshape data into 3D Fortran array ordering
    # -------------------------------------------------------------------------

    npDataChunk = numpy_support.vtk_to_numpy(dataChunk)
    scalars_array3d = np.reshape(npDataChunk, (xSizes[idx], sizeY, sizeZ), order='F')

    print('%d # Data chunk reshaped' % idx)

    # -------------------------------------------------------------------------
    # Reconstruction helper
    # -------------------------------------------------------------------------

    def parallelRay(Nside, pixelWidth, angles, Nray, rayWidth):
        # Suppress warning messages that pops up when dividing zeros
        np.seterr(all='ignore')
        Nproj = len(angles) # Number of projections

        # Ray coordinates at 0 degrees.
        offsets = np.linspace(-(Nray * 1.0 - 1) / 2,
                              (Nray * 1.0 - 1) / 2, Nray) * rayWidth
        # Intersection lines/grid Coordinates
        xgrid = np.linspace(-Nside * 0.5, Nside * 0.5, Nside + 1) * pixelWidth
        ygrid = np.linspace(-Nside * 0.5, Nside * 0.5, Nside + 1) * pixelWidth
        # Initialize vectors that contain matrix elements and corresponding
        # row/column numbers
        rows = np.zeros(2 * Nside * Nproj * Nray)
        cols = np.zeros(2 * Nside * Nproj * Nray)
        vals = np.zeros(2 * Nside * Nproj * Nray)
        idxend = 0

        for i in range(0, Nproj): # Loop over projection angles
            ang = angles[i] * np.pi / 180.
            # Points passed by rays at current angles
            xrayRotated = np.cos(ang) * offsets
            yrayRotated = np.sin(ang) * offsets
            xrayRotated[np.abs(xrayRotated) < 1e-8] = 0
            yrayRotated[np.abs(yrayRotated) < 1e-8] = 0

            a = -np.sin(ang)
            a = rmepsilon(a)
            b = np.cos(ang)
            b = rmepsilon(b)

            for j in range(0, Nray): # Loop rays in current projection
                #Ray: y = tx * x + intercept
                t_xgrid = (xgrid - xrayRotated[j]) / a
                y_xgrid = b * t_xgrid + yrayRotated[j]

                t_ygrid = (ygrid - yrayRotated[j]) / b
                x_ygrid = a * t_ygrid + xrayRotated[j]
                # Collect all points
                t_grid = np.append(t_xgrid, t_ygrid)
                xx = np.append(xgrid, x_ygrid)
                yy = np.append(y_xgrid, ygrid)
                # Sort the coordinates according to intersection time
                I = np.argsort(t_grid)
                xx = xx[I]
                yy = yy[I]

                # Get rid of points that are outside the image grid
                Ix = np.logical_and(xx >= -Nside / 2.0 * pixelWidth,
                                    xx <= Nside / 2.0 * pixelWidth)
                Iy = np.logical_and(yy >= -Nside / 2.0 * pixelWidth,
                                    yy <= Nside / 2.0 * pixelWidth)
                I = np.logical_and(Ix, Iy)
                xx = xx[I]
                yy = yy[I]

                # If the ray pass through the image grid
                if (xx.size != 0 and yy.size != 0):
                    # Get rid of double counted points
                    I = np.logical_and(np.abs(np.diff(xx)) <=
                                       1e-8, np.abs(np.diff(yy)) <= 1e-8)
                    I2 = np.zeros(I.size + 1)
                    I2[0:-1] = I
                    xx = xx[np.logical_not(I2)]
                    yy = yy[np.logical_not(I2)]

                    # Calculate the length within the cell
                    length = np.sqrt(np.diff(xx)**2 + np.diff(yy)**2)
                    #Count number of cells the ray passes through
                    numvals = length.size

                    # Remove the rays that are on the boundary of the box in the
                    # top or to the right of the image grid
                    check1 = np.logical_and(b == 0, np.absolute(
                        yrayRotated[j] - Nside / 2 * pixelWidth) < 1e-15)
                    check2 = np.logical_and(a == 0, np.absolute(
                        xrayRotated[j] - Nside / 2 * pixelWidth) < 1e-15)
                    check = np.logical_not(np.logical_or(check1, check2))

                    if np.logical_and(numvals > 0, check):
                        # Calculate corresponding indices in measurement matrix
                        # First, calculate the mid points coord. between two
                        # adjacent grid points
                        midpoints_x = rmepsilon(0.5 * (xx[0:-1] + xx[1:]))
                        midpoints_y = rmepsilon(0.5 * (yy[0:-1] + yy[1:]))
                        #Calculate the pixel index for mid points
                        pixelIndicex = \
                            (np.floor(Nside / 2.0 - midpoints_y / pixelWidth)) * \
                            Nside + (np.floor(midpoints_x /
                                              pixelWidth + Nside / 2.0))
                        # Create the indices to store the values to the measurement
                        # matrix
                        idxstart = idxend
                        idxend = idxstart + numvals
                        idx = np.arange(idxstart, idxend)
                        # Store row numbers, column numbers and values
                        rows[idx] = i * Nray + j
                        cols[idx] = pixelIndicex
                        vals[idx] = length
                else:
                    print("Ray No. %d at %f degree is out of image grid!" %
                          (j + 1, angles[i]))

        # Truncate excess zeros.
        rows = rows[:idxend]
        cols = cols[:idxend]
        vals = vals[:idxend]
        A = ss.coo_matrix((vals, (rows, cols)), shape=(Nray * Nproj, Nside**2))
        return A


    def rmepsilon(input):
        if (input.size > 1):
            input[np.abs(input) < 1e-10] = 0
        else:
            if np.abs(input) < 1e-10:
                input = 0
        return input


    # -------------------------------------------------------------------------
    # Reconstruction
    # -------------------------------------------------------------------------

    print('%d # Start reconstruction' % idx)

    tiltSeries = scalars_array3d
    tiltAngles = range(-sizeZ + 1, sizeZ, 2) # Delta angle of 2
    (Nslice, Nray, Nproj) = tiltSeries.shape
    Niter = 1

    A = parallelRay(Nray, 1.0, tiltAngles, Nray, 1.0) # A is a sparse matrix
    recon = np.empty([Nslice, Nray, Nray], dtype=float, order='F')

    A = A.todense()

    (Nrow, Ncol) = A.shape
    rowInnerProduct = np.zeros(Nrow)
    row = np.zeros(Ncol)
    f = np.zeros(Ncol) # Placeholder for 2d image
    beta = 1.0

    # Calculate row inner product
    for j in range(Nrow):
        row[:] = A[j, ].copy()
        rowInnerProduct[j] = np.dot(row, row)

    for s in range(Nslice):
        f[:] = 0
        b = tiltSeries[s, :, :].transpose().flatten()
        for i in range(Niter):
            for j in range(Nrow):
                row[:] = A[j, ].copy()
                row_f_product = np.dot(row, f)
                a = (b[j] - row_f_product) / rowInnerProduct[j]
                f = f + row * a * beta

        recon[s, :, :] = f.reshape((Nray, Nray))

    (iSize, jSize, kSize) = recon.shape
    print('%d # End reconstruction - Dimensions (%d, %d, %d)' % (idx, iSize, jSize, kSize))

    # -------------------------------------------------------------------------
    # Convert reconstruction array into VTK format
    # -------------------------------------------------------------------------

    print('%d # Reshape reconstruction' % idx)

    arr = recon.ravel(order='A')
    vtkarray = numpy_support.numpy_to_vtk(arr)
    vtkarray.SetName('Scalars')

    print('%d # Array size %d' % (idx, vtkarray.GetNumberOfTuples()))

    # -------------------------------------------------------------------------
    # Data access helper
    # -------------------------------------------------------------------------

    def createSlice():
        print('%d # createSlice' % (idx))
        size = sizeY * sizeY
        array = np.arange(size, dtype=np.int32)
        print('%d # createSlice => DONE' % (idx))
        return array
    
    def getSideSlice(offset, xSize):
        print('%d # getSideSlice(%d)' % (idx, offset))
        size = sizeY * sizeY
        slice = np.arange(size, dtype=np.int32)
        
        for i in range(size):
            slice[i] = vtkarray.GetTuple1(int(offset + (i * xSize)))
        print('%d # getSideSlice(%d) => DONE' % (idx, offset))
        return slice
        
    # -------------------------------------------------------------------------
    # Add ghost points from neighbors
    # -------------------------------------------------------------------------

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    remoteLowerSlice = None
    remoteUpperSlice = None
    
    if idx + 1 < targetPartition:
        print('%d # Share upper slice' % (idx))
        # Share upper slice
        remoteUpperSlice = createSlice()
        localUpperSlice = getSideSlice(xSizes[idx] - 1, xSizes[idx])
        comm.Sendrecv(localUpperSlice, (idx+1), (2*idx + 1), remoteUpperSlice, (idx+1), (2*idx))
        print('%d # Share upper slice => DONE' % (idx))
    if idx > 0:
        print('%d # Share lower slice' % (idx))
        # Share lower slice
        remoteLowerSlice = createSlice()
        localLowerSlice = getSideSlice(0, xSizes[idx])
        comm.Sendrecv(localLowerSlice, (idx-1), (2 * (idx - 1)), remoteLowerSlice, (idx-1), (2 * (idx - 1) + 1))
        print('%d # Share lower slice => DONE' % (idx))

    print('%d # All MPI done -------' % (idx))
    # -------------------------------------------------------------------------

    print('%d # Create new image data from reconstruction' % idx)

    dataset = vtkImageData()
    minX = 0
    maxX = 0
    for i in range(idx + 1):
        minX = maxX
        maxX += xSizes[i]

    print('%d # extent [%d, %d, %d, %d, %d, %d]' % (idx, minX, maxX - 1, 0, sizeY - 1, 0, sizeY - 1))

    # -------------------------------------------------------------------------
    # Add slice(s) to data
    # -------------------------------------------------------------------------

    arrayWithSlices = vtkarray.NewInstance()
    arrayWithSlices.SetName('Scalars')
    if remoteLowerSlice != None and remoteUpperSlice != None: 
        print('%d # create new array with both slices' % (idx))
        # Add both slices
        minX -= 1
        maxX += 1
        localSizeX = maxX - minX
        newSize = localSizeX * sizeY * sizeY
        arrayWithSlices.SetNumberOfTuples(newSize)
        localOffset = 0
        for i in range(newSize):
            if i % localSizeX == 0:
                arrayWithSlices.SetTuple1(i, remoteLowerSlice[i / localSizeX])
            elif (i + 1) % localSizeX == 0:
                arrayWithSlices.SetTuple1(i, remoteUpperSlice[((i + 1) / localSizeX) - 1])
            else:
                arrayWithSlices.SetTuple1(i, vtkarray.GetTuple1(localOffset))
                localOffset += 1
        print('%d # create new array with both slices => DONE' % (idx))
                
    elif remoteLowerSlice != None:
        print('%d # create new array with lower slices' % (idx))
        # Add lower slice
        minX -= 1
        localSizeX = maxX - minX
        newSize = localSizeX * sizeY * sizeY
        arrayWithSlices.SetNumberOfTuples(newSize)
        localOffset = 0
        for i in range(newSize):
            if i % localSizeX == 0:
                arrayWithSlices.SetTuple1(i, remoteLowerSlice[i / localSizeX])
            else:
                arrayWithSlices.SetTuple1(i, vtkarray.GetTuple1(localOffset))
                localOffset += 1
        print('%d # create new array with lower slices => DONE' % (idx))
    elif remoteUpperSlice != None:
        print('%d # create new array with upper slices' % (idx))
        # Add upper slice
        maxX += 1
        localSizeX = maxX - minX
        newSize = localSizeX * sizeY * sizeY
        arrayWithSlices.SetNumberOfTuples(newSize)
        localOffset = 0
        for i in range(newSize):
            if (i + 1) % localSizeX == 0:
                arrayWithSlices.SetTuple1(i, remoteUpperSlice[((i + 1) / localSizeX) - 1])
            else:
                arrayWithSlices.SetTuple1(i, vtkarray.GetTuple1(localOffset))
                localOffset += 1
        print('%d # create new array with upper slices => DONE' % (idx))

    dataset.SetExtent(minX, maxX - 1, 0, sizeY - 1, 0, sizeY - 1)
    dataset.GetPointData().SetScalars(arrayWithSlices)

    # -------------------------------------------------------------------------  

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

    # -------------------------------------------------------------------------

    print('%d # Start visualization' % idx)

    # -------------------------------------------------------------------------

    args = Options()
    if pm.GetPartitionId() == 0:
        producer = simple.DistributedTrivialProducer()
        producer.UpdateDataset = ''
        producer.UpdateDataset = 'Spark'
        print('Whole extent [%d, %d, %d, %d, %d, %d]' % (0, sizeX - 1, 0, sizeY - 1, 0, sizeY - 1))
        producer.WholeExtent = [0, sizeX - 1, 0, sizeY - 1, 0, sizeY - 1]
        server.start_webserver(options=args, protocol=_VisualizerServer)
        pm.GetGlobalController().TriggerBreakRMIs()

    yield (idx, targetPartition)

# -------------------------------------------------------------------------

results = rdd2.mapPartitionsWithIndex(processPartition).collect()
for out in results:
    print(out)

