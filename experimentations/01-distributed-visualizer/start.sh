#!/usr/bin/env bash

export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/"
export SPARK_HOME="/data/sebastien/SparkMPI/spark-2.1.1-bin-hadoop2.7"

export HYDRA_PROXY_PORT=55555
/data/sebastien/SparkMPI/spark-mpi/install/bin/pmiserv -n 4 hello &

/data/sebastien/SparkMPI/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://beast:7077 ./pvw-spark.py

