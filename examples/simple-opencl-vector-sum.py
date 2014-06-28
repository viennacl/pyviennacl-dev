#!/usr/bin/python3

import pyviennacl as p
import pyopencl as cl
import numpy as np

size = 131073

print("Initialising two random vectors v and w of size %d" % size)

v = np.random.rand(size).astype(np.float32)
w = np.random.rand(size).astype(np.float32)

v = p.Vector(v)
w = p.Vector(w)
x = p.Vector(size, dtype=v.dtype)

ctx = v.context.opencl_context
queue = v.context.current_queue

prg = cl.Program(ctx, """
__kernel void sum(__global const float *a,
                  __global const float *b,
                  __global float *c) {
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
}
""").build()

print("Computing x = v + w using a custom kernel")

prg.sum(queue, v.shape, None,
        v.handle.buffer,
        w.handle.buffer,
        x.handle.buffer)
queue.finish()

print("Done!")

print("Euclidean norm of the difference between x and v+w (computed using ViennaCL): %s" % (x-(v+w)).norm(2))

