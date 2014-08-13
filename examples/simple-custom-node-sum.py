#!/usr/bin/python3

import pyopencl as cl
import numpy as np

from pyviennacl import Vector, Matrix, CustomNode
from pyviennacl.backend import OpenCLMemory

vector_src = """
__kernel void sum(__global const float *a, unsigned int a_size,
                  __global const float *b, unsigned int b_size,
                  __global float *c, unsigned int c_size) {
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
}
"""

matrix_src = """
__kernel void sum(__global const float *a,
                  unsigned int a_size1, unsigned int a_size2,
                  __global const float *b,
                  unsigned int b_size1, unsigned int b_size2,
                  __global float *c,
                  unsigned int c_size1, unsigned int c_size2) {
  int i = get_global_id(0);
  int j = get_global_id(1);
  c[i * c_size1 + j] = a[i * a_size1 + j] + b[i * b_size1 + j];
}
"""

class CustomSum(CustomNode):
    result_types = { ('Vector', 'Vector'): Vector,
                     ('Matrix', 'Matrix'): Matrix }

    kernels = { OpenCLMemory: { ('Vector', 'Vector'): vector_src,
                                ('Matrix', 'Matrix'): matrix_src } }

    @property
    def shape(self): return self.operands[0].shape

    opencl_global_size = shape
    opencl_local_size = None


size = 1025

print("Initialising two random vectors v and w of size %d" % size)

v = np.random.rand(size).astype(np.float32)
w = np.random.rand(size).astype(np.float32)

v = Vector(v)
w = Vector(w)

print("Initialising two random matrices A and B of shape (%d, %d)"
      % (size, size))

A = np.random.rand(size,size).astype(np.float32)
B = np.random.rand(size,size).astype(np.float32)

A = Matrix(A)
B = Matrix(B)

print("Representing x=v+w using custom kernel")

x = CustomSum(v, w)
print(x.express())

print("Representing C=A+B using custom kernel")
C = CustomSum(A, B)
print(C.express())

print("Euclidean norm of the difference between x and v+w (computed using ViennaCL): %s" % (x-(v+w)).norm(2))
print("Frobenius norm of the difference between C and A+B (computed using ViennaCL): %s" % (C-(A+B)).norm('fro'))
print("Done")

