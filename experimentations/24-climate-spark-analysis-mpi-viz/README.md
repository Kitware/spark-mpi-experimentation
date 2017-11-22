To run the visualization, you will have to provide how many Spark processes as well as
how many MPI processes you want to do run on.  The spark processes will be used for reading
the data, summing the daily temperatures for each year, and computing the average high
temperature image for each year.  Then the MPI processes will be used for the visualization

```sh
$ ./start.sh 24 4
```
