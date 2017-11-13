# Introduction

This document aims to describe the spark/mpi processing pipeline built
in this directory ("14-spark-pipeline/pvw-spark.py").  The goal of the
pipeline is to take a set of images made by a 3D scanner, reconstruct a
3D volume from them, and then allow custom parallel visualizations to
be created using the reconstructed volume as input.

## Processing steps

The following ordered list of operations is intended accompany the
"Spark-MPI-demo.pdf", which can be found [here](https://drive.google.com/a/kitware.com/file/d/0B0nURBixaI44emtqQUhENGY2ZFk/view?usp=sharing).

Each numbered operation in this document corresponds to one of the vertical
ellipsoids in the pdf document.  Most of that entire diagram (excluding
reading the original tiff image into a numpy array) can be summarized by
the following few lines of code, which can be found at the bottom of the
`pvw-spark.py` file:

```python
data = sc.parallelize(npArray)
rdd = data.zipWithIndex()                              \
          .map(swap)

rdd.partitionBy(nbSparkPartition, getSparkPartition)   \
   .mapPartitionsWithIndex(reconstruct)                \
   .map(threshold)                                     \
   .partitionBy(nbMPIPartition, getMPIPartition)       \
   .mapPartitionsWithIndex(visualization)              \
   .collect()
```

1. Read

   Reads the tif image "TiltSeries_NanoParticle_doi_10.1021-nl103400a.tif"
   into a Numpy array


   ```python
   filePath = '/data/sebastien/SparkMPI/data-convert/TiltSeries_NanoParticle_doi_10.1021-nl103400a.tif'
   reader = simple.TIFFSeriesReader(FileNames=[filePath])
   reader.UpdatePipeline()
   imageData = reader.GetClientSideObject().GetOutputDataObject(0)
   dataArray = imageData.GetPointData().GetScalars()
   npArray = np.arange(dataArray.GetNumberOfTuples(), dtype=float)
   for idx in range(dataArray.GetNumberOfTuples()):
       npArray[idx] = dataArray.GetTuple1(idx)
   ```

2. parallelize

   Uses the SparkContext to "Parallelize" the numpy array, i.e. turn the array
   into a distributed dataset that can be operated on in parallel.

   ```python
   data = sc.parallelize(npArray)
   ```

   This gives us an `RDD` (or "Resilient Distributed Dataset"), the primary data
   abstraction in Apache Spark.

3. zipWithIndex

   RDD method `zipWithIndex` returns a new RDD in which each element of the input
   RDD has been turned into a tuple containing both the data element and the index
   of the data element in the original RDD.  For example:

   ```
   >>> data = [1, 2, 3, 4]
   >>> rdd = sc.parallelize(data)
   >>> rdd.collect()
   [1, 2, 3, 4]
   >>> rdd.zipWithIndex().collect()
   [(1, 0), (2, 1), (3, 2), (4, 3)]
   >>>
   ```

4. map(swap)

   Next each element in the RDD is run through the `swap` function, which is
   extremely simple.  It takes a tuple of size 2 and returns a new tuple with
   the two elements reversed.  In this way, subsequent processing functions
   will receive a tuple where the first element is the index and the second
   element is a piece of the original data from the tif image.

   To see the result of the `zip` and `map(swap)` operations, we can pick up
   where the last pyspark interpreter session left off:

   ```python
   def swap(kv):
       return (kv[1], kv[0])
   ```

   For example, picking up from where the last pyspark interpreter session left off:

   ```
   >>> def swap(kv):
   ...   return (kv[1], kv[0])
   ...
   >>> rdd.zipWithIndex().map(swap).collect()
   [(0, 1), (1, 2), (2, 3), (3, 4)]
   >>>
   ```

   This output corresponds to the first RDD dataset block in the diagram with
   (blue, green) database icons.

5. partitionBy(nbSparkPartition, getSparkPartition)

   The arguments to `partitionBy` at this stage in the processing are the number
   of spark partitions (given to the start script on the command line) as well as
   a function that can assign each element (tuple) to one of those partitions.
   The result of this function is that the data is partitioned into blocks, such
   that each spark process can be handed a block of it.

6. mapPartitionsWithIndex(reconstruct)

   The `reconstruct` method takes a partition index and an iterator method.  The
   iterator method is used to iterate over all the elements, in this case tuples
   of (idx, data), in a partition/block of data.  The `reconstruct` method then
   yields up new tuples in the form (gIdx, recon[i][j][k]), where the second
   element is a cell from the 3D reconstruction.

7. map(threshold)

   Now the data is passed, element (tuple) by element, through the threshold
   function where small or negative numbers (anything less than 2.0) are simply
   replaced by 0.0.  The same tuple shape of (idx, data) is maintained.

8. partitionBy(nbMPIPartition, getMPIPartition)

   At this point the data is repartitioned to match the number of MPI processes
   given on the command line to the start script.  The partitioning function in
   this case behaves in precisely the same way as the one used to partition the
   data for spark processes, i.e. it returns a number in the range
   `[0, npMPIPartition - 1]` to indicate which mpi process should handle each
   tuple.  The index arithmetic is done such that each process will get a
   contiguous chunk of the reconstructed 3D volume.

9. mapPartitionsWithIndex(visualization)

   In this step the mapping function, `visualization` will operate on an entire
   data block (as partitioned by the previous step), so its arguments are the
   partition index and an iterator used to step through the elements in the
   block.  The job of the `visualization` function is to create a vtkImageData
   corresponding to the slice of the reconstructed 3D volume it was given to
   render, adding ghost cells on one or both sides of the slice, as necessary.
   Then the function indicates the extent of its slice within the larger global
   dataset and sets the vtkImageData into the global output of a trivial producer.
   If the function happens to get assigned partition index 0, then it additionally
   sets the WholeExtent of the producer and starts the ParaViewWeb server process
   allowing a client to connect and create a visualization pipeline.

