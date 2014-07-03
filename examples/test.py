import pyviennacl as p
import numpy as np

from pyviennacl.backend import OpenCLMemory, Context
ctx = Context(OpenCLMemory)

a = np.ones(100).astype(np.float32)
a = p.Vector(a, context=ctx)

b = p.CompressedMatrix(context=ctx, dtype=np.float32)
b[1,2] = 3.0
b.flush()

a = np.ones(1024).astype(np.float32)
a = p.Vector(a, context=ctx)

print(a)
