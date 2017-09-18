import os, gdal
from datetime import datetime

CHUNK_SIZE = 10000

# ---------------------------------------------------------------------------

def chunkReader(fileName):
    year = fileName.split('_')[-1][:-4]
    print('year', year)

    dataset = gdal.Open(fileName)
    imgDims = [dataset.RasterYSize, dataset.RasterXSize]
    numPixels = imgDims[0] * imgDims[1]
    loopCount = numPixels / CHUNK_SIZE
    leftOverChunkSize = numPixels % CHUNK_SIZE

    print('There will be %d chunks with %d leftover' % (loopCount, leftOverChunkSize))

    for bandId in range(dataset.RasterCount):
        print('NEXT BAND: %d' % (bandId + 1))
        band = dataset.GetRasterBand(bandId + 1).ReadAsArray().flatten()

        for i in range(loopCount):
            yield ('%s-%d' % (year, i), list(band[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]))

        yield ('%s-%d' % (year, loopCount), list(band[loopCount*CHUNK_SIZE:]))

# ---------------------------------------------------------------------------

filePath = '/data/scott/SparkMPI/data/gddp/tasmax_day_BCSD_rcp85_r1i1p1_MRI-CGCM3_2006.tif'

for chunk in chunkReader(filePath):
    if chunk[0] == '2006-96':
        print('next chunk: %s, size: %d' % (chunk[0], len(chunk[1])))
        print(chunk[1][:10])

