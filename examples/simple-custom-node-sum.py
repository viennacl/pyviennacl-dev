#!/usr/bin/python3

import pyopencl as cl
import numpy as np

from pyviennacl import Vector, CustomNode
from pyviennacl.backend import OpenCLMemory

size = 131073

print("Initialising two random vectors v and w of size %d" % size)

v = np.random.rand(size).astype(np.float32)
w = np.random.rand(size).astype(np.float32)

v = Vector(v)
w = Vector(w)

print("Representing x=v+w using custom kernel")

src = """
__kernel void sum(__global const float *a,
                  __global const float *b,
                  __global float *c) {
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
}
"""

class CustomSum(CustomNode):
    result_types = { ('Vector', 'Vector'): Vector }

    kernels = { OpenCLMemory: {
            ('Vector', 'Vector'): src
        } }

x = CustomSum(v, w)
print(x.express())

print("Euclidean norm of the difference between x and v+w (computed using ViennaCL): %s" % (x-(v+w)).norm(2))
print("Done")

