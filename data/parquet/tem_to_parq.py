#!/usr/bin/env python

# -*- coding: utf-8 -*-

# Imports
from skimage import io

import pandas as pd

import fastparquet as fp

import numpy as np

import os

import tempfile

def readTiff(filename):
    """Read data from the tiff file and return a Pandas dataframe"""
    filenamePrefix = os.path.splitext(os.path.basename(filename))[0]

    im = io.imread(filename)

    # Reshape 3D to one giant 2D
    imgdata2d = im.reshape(im.shape[0], im.shape[1] * im.shape[2])
    index = range(0, im.shape[0])

    # Convert to Pandas dataframe
    df = pd.DataFrame(imgdata2d.astype(np.int32),
                      columns=[str(i) for i in range(0, im.shape[1] * im.shape[2])],
                      index=index)

    # TODO Optimize by squeezing or by finding another way
    # of adding metdata to the files (Perhaps we can store
    # metadata into a separate parquet file)
    df['tiltangles'] = pd.Series([i%df.shape[0] for i in
                         range(0, df.shape[0], 1)], index=df.index)
    df['dimensions'] = pd.Series([im.shape[0], im.shape[1], im.shape[2]]
                         for i in range(0, df.shape[0], 1))
    df['pixelsize'] = pd.Series(10 for i in range(0, df.shape[0], 1))

    return df


def writeParquet(inputFilename, df):
    """Export Pandas dataframe as Parquet"""
    filenamePrefix = os.path.splitext(os.path.basename(inputFilename))[0]
    outFilepath = os.path.join(tempfile.gettempdir(), ''.join([filenamePrefix, '.parq']))
    fp.write(outFilepath, df, compression='GZIP')
    print outFilepath	
    return outFilepath


def uploadToHDFS(hdfsroot, filepath):
    """Upload data to HDFS and delete the temp file"""
    filename = os.path.basename(filepath)
    from subprocess import call

    command = ' '.join(["hadoop fs -put", filepath, os.path.join(hdfsroot, filename)])

    print command

    errcode = call(command, shell=True)

    # Remove the file if the upload succeeds
    if errcode == 0:
        os.remove(filepath)
    else:
        print ("failed to upload file %s to HDFS %s" % (filepath, hdfsroot))


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Script to convert data in tiff format to Parquet format')
    parser.add_argument('--tiff', dest='filename', help='input tiff file')
    parser.add_argument('--hdfs', dest='hdfsroot', help='HDFS root for data upload')
    args = parser.parse_args()

    # Read TIFF file and convert it into a Pandas dataframe
    df = readTiff(args.filename)

    # Export dataframe as parquet
    outFilepath = writeParquet(args.filename, df)

    # Upload to HDFS if HDFS root provided
    if args.hdfsroot:
        uploadToHDFS(args.hdfsroot, outFilepath)
