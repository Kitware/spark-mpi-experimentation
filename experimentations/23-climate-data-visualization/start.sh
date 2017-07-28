#!/usr/bin/env bash

export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/"
export SPARK_HOME="/data/scott/SparkMPI/spark-2.1.1-bin-hadoop2.7"
export MPI_SIZE=$1

export HYDRA_PROXY_PORT=55555

/data/scott/SparkMPI/spark-mpi/install/bin/pmiserv -n ${MPI_SIZE} hello &

/data/scott/SparkMPI/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://beast:7077 --driver-memory 20g  --executor-memory 10g ./pvw-spark.py

