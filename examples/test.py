from timeit import timeit

setup = """
import pyviennacl as p
import numpy as np

from pyviennacl.backend import OpenCLMemory, Context
ctx = Context(OpenCLMemory)

a = np.random.rand(1000).astype(np.float32)
"""

stmt = "a = p.Vector(a, context=ctx)"

time = timeit(stmt, setup=setup, number=20) / 20

#b = p.CompressedMatrix(context=ctx, dtype=np.float32)
#b[1,2] = 3.0
#b.flush()


print(time)
