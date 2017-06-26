# Spark-MPI experimentation

This repository gather various experimentation made with Spark along with MPI parallel processing and ParaView.

## Configuration of the test machine

### Working directory

```
mkdir -p /data/sebastien/SparkMPI
cd /data/sebastien/SparkMPI
```

### Get ParaView

```
mkdir pv-build pv-install mpi
git clone https://gitlab.kitware.com/paraview/paraview.git
cd paraview
git submodule update --init
```

### Get CMake

```
mkdir cmake
cd cmake
curl -O https://cmake.org/files/v3.8/cmake-3.8.1-Linux-x86_64.sh
chmod +x cmake-3.8.1-Linux-x86_64.sh
./cmake-3.8.1-Linux-x86_64.sh
```

### Get MPI library to use

```
cd mpi
curl -O http://mvapich.cse.ohio-state.edu/download/mvapich/mv2/mvapich2-2.2.tar.gz
tar xvfz mvapich2-2.2.tar.gz
mkdir mvapich
cd mvapich2-2.2
./configure --disable-libxml2 --disable-fortran --prefix=/data/sebastien/SparkMPI/mpi/mvapich  --disable-mcast  --without-cma
make
make install
```

### Build ParaView with MPI

```
cd /data/sebastien/SparkMPI/pv-build
/data/sebastien/SparkMPI/cmake/cmake-3.8.1-Linux-x86_64/bin/ccmake ../paraview

CMAKE_INSTALL_PREFIX            */data/sebastien/SparkMPI/pv-install/
PARAVIEW_BUILD_QT_GUI           *OFF
PARAVIEW_ENABLE_PYTHON          *ON
PARAVIEW_USE_MPI                *ON

[c]

MPI_C_INCLUDE_PATH              */data/sebastien/SparkMPI/mpi/mvapich/include
MPI_C_LIBRARIES                 */data/sebastien/SparkMPI/mpi/mvapich/lib/libmpi.so

[c]
[g]

make -j20
make install
```

### Install Spark-MPI

```
export MPI_SRC=/data/sebastien/SparkMPI/mpi/mvapich2-2.2/src/

mkdir -p /data/sebastien/SparkMPI/spark-mpi
cd /data/sebastien/SparkMPI/spark-mpi/
git clone git://github.com/SciDriver/spark-mpi.git
mkdir build install
cd build
/data/sebastien/SparkMPI/cmake/cmake-3.8.1-Linux-x86_64/bin/cmake ../spark-mpi

CMAKE_INSTALL_PREFIX            */data/sebastien/SparkMPI/spark-mpi/install
MPI_EXTRA_LIBRARY               */data/sebastien/SparkMPI/mpi/mvapich/lib/libmpi.so
MPI_LIBRARY                     */data/sebastien/SparkMPI/mpi/mvapich/lib/libmpicxx.so

MPIEXEC                         */data/sebastien/SparkMPI/mpi/mvapich/bim/mpiexec
MPIEXEC_MAX_NUMPROCS            *2
MPIEXEC_NUMPROC_FLAG            *-np
MPIEXEC_POSTFLAGS               *
MPIEXEC_PREFLAGS                *
MPI_CXX_COMPILER                */data/sebastien/SparkMPI/mpi/mvapich/bin/mpicxx
MPI_CXX_COMPILE_FLAGS           *
MPI_CXX_INCLUDE_PATH            */data/sebastien/SparkMPI/mpi/mvapich/include
MPI_CXX_LIBRARIES               */data/sebastien/SparkMPI/mpi/mvapich/lib/libmpicxx.so;/data/sebastien/SparkMPI/mpi/mvapich/lib/libmpi.so
MPI_CXX_LINK_FLAGS              *-Wl,-rpath -Wl,/data/sebastien/SparkMPI/mpi/mvapich/lib -Wl,--enable-new-dtags
MPI_C_COMPILER                  */data/sebastien/SparkMPI/mpi/mvapich/bin/mpicc
MPI_C_COMPILE_FLAGS             *
MPI_C_INCLUDE_PATH              */data/sebastien/SparkMPI/mpi/mvapich/include
MPI_C_LIBRARIES                 */data/sebastien/SparkMPI/mpi/mvapich/lib/libmpi.so
MPI_C_LINK_FLAGS                *-Wl,-rpath -Wl,/usr/lib/openmpi/lib -Wl,--enable-new-dtags

CMAKE_BUILD_TYPE                 Release

[c][g]

make
make install
```

### Install Spark

```
curl -O http://d3kbcqa49mib13.cloudfront.net/spark-2.1.1-bin-hadoop2.7.tgz
tar xvfz spark-2.1.1-bin-hadoop2.7.tgz

export SPARK_HOME=/data/sebastien/SparkMPI/spark-2.1.1-bin-hadoop2.7
cd $SPARK_HOME/conf
cp spark-defaults.conf.template spark-defaults.conf
vi spark-defaults.conf

  spark.driver.memory    5g

cp slaves.template slaves

vi spark-env.sh

export PYTHONPATH="${PYTHONPATH}:/data/sebastien/SparkMPI/pv-install/lib/paraview-5.4/site-packages/vtk"
export PYTHONPATH="${PYTHONPATH}:/data/sebastien/SparkMPI/pv-install/lib/paraview-5.4/site-packages"
export PYTHONPATH="${PYTHONPATH}:/data/sebastien/SparkMPI/pv-install/lib/paraview-5.4"
export PYTHONPATH="${PYTHONPATH}:/data/sebastien/SparkMPI/pv-install/bin"
export LD_LIBRARY_PATH=/data/sebastien/SparkMPI/pv-install/lib/paraview-5.4:/data/sebastien/SparkMPI/pv-install/lib/paraview-5.4/site-packages/vtk
```

Start spark

```
cd /data/sebastien/SparkMPI/spark-2.1.1-bin-hadoop2.7/sbin
./start-all.sh
```

### Install SciPy

__Atlas library__

```
$ sudo apt-get install libatlas-base-dev
```

__BLAS library__

```
$ sudo apt-get install libopenblas-dev
```

__MKL library__

Register on Intel web site
=> https://software.intel.com/en-us/performance-libraries

Then download MKL (following link may not work for you)

```
$ wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/11544/l_mkl_2017.3.196.tgz
$ tar xvfz l_mkl_2017.3.196.tgz
$ cd l_mkl_2017.3.196/
$ sudo ./install.sh
```

```
Install location:
    /opt/intel

Component(s) selected:
    Intel(R) Math Kernel Library 2017 Update 3 for C/C++                   2.0GB
        Intel MKL core libraries for C/C++
        Intel TBB threading support
        GNU* C/C++ compiler support

    Intel(R) Math Kernel Library 2017 Update 3 for Fortran                 2.1GB
        Intel MKL core libraries for Fortran
        GNU* Fortran compiler support
        Fortran 95 interfaces for BLAS and LAPACK

Install space required:  2.3GB
```

(from: http://tzutalin.blogspot.com/2015/06/blas-atlas-openblas-and-mkl.html)

=> create file ~/.numpy-site.cfg

```
[DEFAULT]
library_dirs = /usr/lib:/usr/local/lib
include_dirs = /usr/include:/usr/local/include

[mkl]
library_dirs = /opt/intel/mkl/lib/intel64/
include_dirs = /opt/intel/mkl/include/
mkl_libs = mkl_intel_ilp64, mkl_intel_thread, mkl_core, mkl_rt
lapack_libs =

[amd]
amd_libs = amd

[umfpack]
umfpack_libs = umfpack

[djbfft]
include_dirs = /usr/local/djbfft/include
library_dirs = /usr/local/djbfft/lib
```

__scipy__

```
$ mkdir scipy
$ cd scipy
$ wget https://github.com/scipy/scipy/releases/download/v0.19.0/scipy-0.19.0.tar.gz
$ tar xvfz scipy-0.19.0.tar.gz
$ cd scipy-0.19.0

$ export PYTHONPATH=/data/sebastien/SparkMPI/scipy/install/lib/python2.7/site-packages
$ python setup.py install --prefix=/data/sebastien/SparkMPI/scipy/install

$ cp -r /data/sebastien/SparkMPI/scipy/install/lib/python2.7/site-packages/scipy-0.19.0-py2.7-linux-x86_64.egg/scipy /data/sebastien/SparkMPI/pv-install/lib/paraview-5.4/site-packages
```

### Patch ParaView Python server

Edit `~/SparkMPI/pv-install/lib/paraview-5.4/site-packages/vtk/web/server.py` and remove all occurence involving the "testing" module.

### Running experimentations

```
$ cd /data/sebastien/SparkMPI
$ git clone https://github.com/Kitware/spark-mpi-experimentation.git
$ cd spark-mpi-experimentation/experimentations
```

Choose the example to run

```
$ cd 11-recon-to-volume
$ ./start.sh
```

__gdal__

```
$ sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
$ sudo apt-get install gdal-bin libgdal-dev

$ export CPLUS_INCLUDE_PATH=/usr/include/gdal
$ export C_INCLUDE_PATH=/usr/include/gdal

$ virtualenv tmp-gdal
$ pip install gdal==2.1.0

$ cd tmp-gdal/lib/python2.7/site-packages
$ cp -r gdal* /data/sebastien/SparkMPI/pv-install/lib/paraview-5.4/site-packages
$ cp -r skimage /data/sebastien/SparkMPI/pv-install/lib/paraview-5.4/site-packages
$ cp -r osgeo /data/sebastien/SparkMPI/pv-install/lib/paraview-5.4/site-packages
```