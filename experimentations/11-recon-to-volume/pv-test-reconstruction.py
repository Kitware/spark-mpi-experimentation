from __future__ import print_function

import os
import sys

from paraview import simple
import vtk
import numpy as np
import scipy.sparse as ss
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support
from vtk.vtkCommonCore import vtkIntArray

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
print('dimensions', originalDimensions)
npArray = numpy_support.vtk_to_numpy(intArray)

scalars_array3d = np.reshape(npArray, (originalDimensions), order='F')

# -------------------------------------------------------------------------

from vtk.vtkCommonExecutionModel import vtkTrivialProducer
from vtk.vtkPVVTKExtensionsCore import vtkDistributedTrivialProducer
from vtk.vtkCommonCore import vtkIntArray, vtkUnsignedCharArray
from vtk.vtkCommonDataModel import vtkImageData, vtkPointData, vtkCellData

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

tiltSeries = scalars_array3d
tiltAngles = range(-73, 74, 2)
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

# -------------------------------------------------------------------------

arr = recon.ravel(order='A')
# arr = recon.reshape(-1, order='F')
vtkarray = numpy_support.numpy_to_vtk(arr)

# -------------------------------------------------------------------------

dataset = vtkImageData()
dataset.SetDimensions(originalDimensions[0], originalDimensions[1], originalDimensions[1])
dataset.GetPointData().SetScalars(vtkarray)

writer = vtk.vtkDataSetWriter()
writer.SetFileName('/tmp/recon.vtk')
writer.SetInputData(dataset)
writer.Write()

print('write file to /tmp/recon.vtk')