#!/usr/bin/env bash

export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64/"
export SPARK_HOME="/data/sebastien/SparkMPI/spark-2.1.1-bin-hadoop2.7"

/data/sebastien/SparkMPI/spark-2.1.1-bin-hadoop2.7/bin/spark-submit --master spark://beast:7077 ./spark-stream-receiver.py

