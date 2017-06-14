from __future__ import print_function

import os
import sys
from pyspark import SparkContext

sc = SparkContext()

# Define the address of the PMI server and the number of MPI workers

hostname = os.uname()[1]
hydra_proxy_port = os.getenv("HYDRA_PROXY_PORT")
pmi_port = hostname + ":" + hydra_proxy_port

partitions = 4

# Prepare a list of environmental variables

env = []
for id in range(0, partitions):
    kvs = {
        'PMI_PORT' : pmi_port,
        'PMI_ID' : id,
    }
    env.append(kvs)

# Create the rdd collection associated with the MPI workers

rdd = sc.parallelize(env, partitions)

# Define the MPI application

def allreduce(kvs):
    import os
    import sys
    os.environ["PMI_PORT"] = kvs["PMI_PORT"]
    os.environ["PMI_ID"] = str(kvs["PMI_ID"])

    from vtk.vtkPVClientServerCoreCore import vtkProcessModule, vtkPVOptions
    from vtk.vtkPVServerManagerApplication import vtkInitializationHelper
    pm = vtkProcessModule.GetProcessModule()

    if not pm:
       pvoptions = vtkPVOptions()
       pvoptions.SetProcessType(vtkPVOptions.PVBATCH)
       vtkInitializationHelper.Initialize(sys.executable, vtkProcessModule.PROCESS_BATCH, pvoptions)
       pm = vtkProcessModule.GetProcessModule()


    out = {
       'rank' : pm.GetPartitionId(),
       'size': pm.GetNumberOfLocalPartitions()
    }
    return out

# Run MPI application on Spark workers and collect the results

results = rdd.map(allreduce).collect()
for out in results:
    print ("rank: ", out['rank'], 'size:', out['size'])
