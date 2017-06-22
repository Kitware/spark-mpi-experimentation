from __future__ import print_function
import os
import sys
import time

import pyspark

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext()
ssc = StreamingContext(sc, 1)

# -------------------------------------------------------------------------
# Stream handling
# -------------------------------------------------------------------------

lines = ssc.socketTextStream("localhost", 65432)

lines.map(lambda s: float(s)).filter(lambda v: v < 5000).map(lambda v: (v, 1)).reduce(lambda a, b: (a[0] + b[0], a[1] + b[1])).pprint()

ssc.start()
ssc.awaitTermination()

                                                                                               
