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

    # Reshape 3D to one giant 1D
    imgdata1d = im.reshape(im.shape[0] * im.shape[1] * im.shape[2])
    dataSize = im.shape[0] * im.shape[1] * im.shape[2]
    sliceSize = im.shape[2] * im.shape[1]

    data = {
        'x': [(i % im.shape[2]) for i in range(0, dataSize)],
        'y': [(i / im.shape[2] % im.shape[1]) for i in range(0, dataSize)],
        'z': [int(i / sliceSize) for i in range(0, dataSize)],
        'value': imgdata1d.astype(np.int32),
    }

    # Convert to Pandas dataframe
    df = pd.DataFrame(data)

    return df


def writeParquet(inputFilename, df):
    """Export Pandas dataframe as Parquet"""
    filenamePrefix = os.path.splitext(os.path.basename(inputFilename))[0]
    outFilepath = os.path.join(tempfile.gettempdir(), ''.join([filenamePrefix, '.parq']))
    fp.write(outFilepath, df, compression='GZIP')
    print outFilepath	
    return outFilepath


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Script to convert data in tiff format to Parquet format')
    parser.add_argument('--tiff', dest='filename', help='input tiff file')
    args = parser.parse_args()

    # Read TIFF file and convert it into a Pandas dataframe
    df = readTiff(args.filename)

    # Export dataframe as parquet
    outFilepath = writeParquet(args.filename, df)
