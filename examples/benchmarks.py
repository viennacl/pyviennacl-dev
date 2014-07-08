#!/usr/bin/env python3

from __future__ import division

#
# Configuration options for benchmarks
#

# Default benchmarks

ADD = True
GEMV = True
SPGEMV = True
GEMM = True
SPGEMM = True

# Platforms

PYVIENNACL = True  # PyViennaCL benchmarks
NUMPY_SCIPY = True # NumPy / SciPy benchmarks
CUDA = True        # Only if gnumpy is installed

# Matrix structure parameters

ADD_SIZES = [int(10**(x/3)) for x in range(12,27)]
GEMV_SIZES = [int(10**(x/4)) for x in range(8,17)]
GEMM_SIZES = [int(10**(x/4)) for x in range(8,17)]
SPGEMV_SIZES = [int(10**(x/4)) for x in range(8,17)]
SPGEMM_SIZES = [int(10**(x/4)) for x in range(8,17)]
SPARSITY = 0.02

################################################################################

import sys, time
import numpy as np
from pyviennacl.backend import Context
from timeit import timeit
try: import gnumpy
except ImportError: CUDA = False
try: import pyopencl as cl
except ImportError: PYVIENNACL = False

def do_benchmark(setup, do_op, sizes, num_iters=5,
                 sparsity=None, cl_device=None):
    if cl_device is None:
        ctx = None
    else:
        ctx = Context(cl.Context([cl_device]))

    for size in sizes:
        try:
            A, B = setup(size, sparsity=sparsity, context=ctx)
            if ctx is not None: ctx.finish_all_queues()
            for i in range(3):
                do_op(A, B, ctx)

            N = 0
            current_time = 0
            if ctx is not None: ctx.finish_all_queues() # Just to be sure
            while current_time < 1:
                time_before = time.time()
                do_op(A, B, ctx)
                current_time += time.time() - time_before
                N += 1
            print(size, current_time / N)
        except Exception as e:
            print("Exception with size %d: %s" % (size, e))
            break

def setup_vector_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    import pyviennacl as p

    x = np.ones(size).astype(dtype) * 0.3
    y = np.ones(size).astype(dtype) * 0.9

    x = p.Vector(x, context=context)
    y = p.Vector(y, context=context)

    return x, y

def setup_gemm_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    import pyviennacl as p

    A = np.ones((size,size)).astype(dtype) * 0.3
    B = np.ones((size,size)).astype(dtype) * 0.9

    A = p.Matrix(A, context=context)
    B = p.Matrix(B, context=context)

    return A, B

def setup_gemv_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    import pyviennacl as p

    A = np.ones((size,size)).astype(dtype) * 0.5
    A = p.Matrix(A, context=context)

    x = np.ones((size,)).astype(dtype) * 2.0
    x = p.Vector(x, context=context)

    return A, x

def setup_spgemv_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    import pyviennacl as p
    import math

    nnz = int(max(1, math.ceil((size*size)*sparsity)))
    mod = nnz

    x = np.ones((size,)).astype(dtype) * 1.1
    x = p.Vector(x, context=context)

    values = np.array([], dtype=dtype)
    max_size = 10**6
    while mod > 0:
        if mod < max_size:
            values = np.append(values, np.ones((mod,)).astype(dtype) * 0.4)
            mod = 0
        else:
            values = np.append(values, np.ones((max_size,)).astype(dtype) * 0.4)
            mod -= max_size
    data = []
    while len(data) < nnz:
        r = np.random.randint(0, size-1)
        c = np.random.randint(0, size-1)
        if (r, c) not in data:
            data.append((r, c))
    data = list(map(list, zip(*data)))
    data.append(values)
    del values
    data = tuple(data)

    A = p.CompressedMatrix(data, shape=(size, size, nnz),
                           dtype=dtype, context=context)
    A.flush()

    return A, x

def setup_spgemm_pyvcl(size, sparsity = None, device = None, dtype = np.float32):
    import pyviennacl as p
    import math

    nnz = int(max(1, math.ceil((size*size)*sparsity)))
    mod = nnz

    values = np.array([], dtype=dtype)
    max_size = 10**6
    while mod > 0:
        if mod < max_size:
            values = np.append(values, np.ones((mod,)).astype(dtype) * 0.25)
            mod = 0
        else:
            values = np.append(values, np.ones((max_size,)).astype(dtype) * 0.25)
            mod -= max_size
    data = []
    while len(data) < nnz:
        r = np.random.randint(0, size-1)
        c = np.random.randint(0, size-1)
        if (r, c) not in data:
            data.append((r, c))
    data = list(map(list, zip(*data)))
    data.append(values)
    del values
    data = tuple(data)

    A = p.CompressedMatrix(data, shape=(size, size, nnz),
                           dtype=dtype, context=context)
    A.flush()

    B = p.CompressedMatrix(data, shape=(size, size, nnz),
                           dtype=dtype, context=context)
    B.flush()

    return A, B

def setup_vector_gnumpy(size, sparsity = None, context = None, dtype = np.float32):
    import gnumpy as gnp

    x = np.ones(size).astype(dtype) * 0.3
    y = np.ones(size).astype(dtype) * 0.9

    x = gnp.garray(x)
    y = gnp.garray(y)

    return x, y

def setup_gemm_gnumpy(size, sparsity = None, context = None, dtype = np.float32):
    import gnumpy as gnp

    A = np.ones((size,size)).astype(dtype) * 0.7
    B = np.ones((size,size)).astype(dtype) * 0.5

    A = gnp.garray(A)
    B = gnp.garray(B)

    return A, B

def setup_gemv_gnumpy(size, sparsity = None, context = None, dtype = np.float32):
    import gnumpy as gnp

    A = np.ones((size,size)).astype(dtype) * 0.3
    A = gnp.garray(A)

    x = np.ones((size,)).astype(dtype) * 0.7
    x = gnp.garray(x)

    return A, x

def setup_vector_numpy(size, sparsity = None, context = None, dtype = np.float32):
    x = np.ones(size).astype(dtype) * 0.3
    y = np.ones(size).astype(dtype) * 0.9

    return x, y

def setup_gemv_numpy(size, sparsity = None, context = None, dtype = np.float32):
    A = np.ones((size,size)).astype(dtype) * 0.8
    x = np.ones((size,)).astype(dtype) * 0.9

    return A, x

def setup_gemm_numpy(size, sparsity = None, context = None, dtype = np.float32):
    A = np.ones((size,size)).astype(dtype) * 0.6
    B = np.ones((size,size)).astype(dtype) * 0.3

    return A, B

def setup_spgemv_scipy(size, sparsity = None, context = None, dtype = np.float32):
    import scipy.sparse as sp
    import math

    nnz = int(math.ceil((size*size)*sparsity))
    mod = nnz

    x = np.ones((size,)).astype(dtype) * 0.2

    values = np.array([], dtype=dtype)
    max_size = 10**6
    while mod > 0:
        if mod < max_size:
            values = np.append(values, np.ones((mod,)).astype(dtype) * 0.8)
            mod = 0
        else:
            values = np.append(values, np.ones((max_size,)).astype(dtype) * 0.8)
            mod -= max_size
    rows = np.random.randint(0, size-1, size=nnz)
    cols = np.random.randint(0, size-1, size=nnz)

    A = sp.coo_matrix((values, (rows, cols)), shape=(size, size), dtype=dtype)

    return A, x


def setup_spgemm_scipy(size, sparsity = None, context = None, dtype = np.float32):
    import scipy.sparse as sp
    import math

    nnz = int(math.ceil((size*size)*sparsity))
    mod = nnz

    values = np.array([], dtype=dtype)
    max_size = 10**6
    while mod > 0:
        if mod < max_size:
            values = np.append(values, np.ones((mod,)).astype(dtype) * 0.6)
            mod = 0
        else:
            values = np.append(values, np.ones((max_size,)).astype(dtype) * 0.6)
            mod -= max_size
    rows = np.random.randint(0, size-1, size=nnz)
    cols = np.random.randint(0, size-1, size=nnz)

    A = sp.coo_matrix((values, (rows, cols)), shape=(size, size), dtype=dtype)
    B = sp.coo_matrix((values, (rows, cols)), shape=(size, size), dtype=dtype)

    return A, B

def add_pyvcl(A, B, ctx = None):
    res = (A+B).execute()
    ctx.finish_all_queues()

def mul_pyvcl(A, B, ctx = None):
    res = (A*B).execute()
    ctx.finish_all_queues()

def gemm_pyvcl(A, B, ctx = None):
    res = (A.T*B).execute()
    ctx.finish_all_queues()

def add_numpy(A, B, ctx = None):
    return A+B

def mul_numpy(A, B, ctx = None):
    return A.dot(B)

def gemm_numpy(A, B, ctx = None):
    return A.T.dot(B)


def main(arg):

    #
    # Vector addition
    #

    if arg == "add":

        print("OPERATION: Vector Addition")

        if PYVIENNACL:
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    print("PLATFORM: PyViennaCL: %s" % device.name)
                    try:
                        do_benchmark(setup_vector_pyvcl, add_pyvcl, ADD_SIZES,
                                     cl_device=device)
                    except KeyboardInterrupt:
                        print(" !!! Interrupted, so moving on...")
                        continue
                    print("")
                
        if NUMPY_SCIPY:
            print("PLATFORM: NumPy")
            try:
                do_benchmark(setup_vector_numpy, add_numpy, ADD_SIZES)
            except KeyboardInterrupt:
                print(" !!! Interrupted, so moving on...")
            print("")

        if CUDA:
            print("PLATFORM: gnumpy")
            try:
                do_benchmark(setup_vector_gnumpy, add_numpy, ADD_SIZES)
            except KeyboardInterrupt:
                print(" !!! Interrupted, so moving on...")
            print("")

    #
    # Dense matrix multiplication
    #
    
    elif arg == "gemm":

        print("OPERATION: Dense Matrix-Matrix Product")

        if PYVIENNACL:
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    print("PLATFORM: PyViennaCL: %s" % device.name)
                    try:
                        do_benchmark(setup_gemm_pyvcl, gemm_pyvcl, GEMM_SIZES,
                                     cl_device=device)
                    except KeyboardInterrupt:
                        print(" !!! Interrupted, so moving on...")
                        continue
                    print("")

        if NUMPY_SCIPY:
            print("PLATFORM: NumPy")
            try:
                do_benchmark(setup_gemm_numpy, gemm_numpy, GEMM_SIZES)
            except KeyboardInterrupt:
                print(" !!! Interrupted, so moving on...")
            print("")
    
        if CUDA:
            print("PLATFORM: gnumpy")
            try:
                do_benchmark(setup_gemm_gnumpy, gemm_numpy, GEMM_SIZES)
            except KeyboardInterrupt:
                print(" !!! Interrupted, so moving on...")
            print("")


    #
    # Dense matrix-vector multiplication
    #

    elif arg == "gemv":

        print("OPERATION: Dense Matrix-Vector Product")

        if PYVIENNACL:
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    print("PLATFORM: PyViennaCL: %s" % device.name)
                    try:
                        do_benchmark(setup_gemv_pyvcl, mul_pyvcl, GEMV_SIZES,
                                     cl_device=device)
                    except KeyboardInterrupt:
                        print(" !!! Interrupted, so moving on...")
                        continue
                    print("")

        if NUMPY_SCIPY:
            print("PLATFORM: NumPy")
            try:
                do_benchmark(setup_gemv_numpy, mul_numpy, GEMV_SIZES)
            except KeyboardInterrupt:
                print(" !!! Interrupted, so moving on...")
            print("")
    
        if CUDA:
            print("PLATFORM: gnumpy")
            try:
                do_benchmark(setup_gemv_gnumpy, mul_numpy, GEMV_SIZES)
            except KeyboardInterrupt:
                print(" !!! Interrupted, so moving on...")
            print("")


    #
    # Sparse matrix-matrix multiplication
    #

    elif arg == "spgemm":

        print("OPERATION: Sparse (%f) Matrix-Matrix Product" % SPARSITY)

        if PYVIENNACL:
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    print("PLATFORM: PyViennaCL: %s" % device.name)
                    try:
                        do_benchmark(setup_spgemm_pyvcl, mul_pyvcl, SPGEMM_SIZES,
                                     sparsity=SPARSITY, cl_device=device)
                    except KeyboardInterrupt:
                        print(" !!! Interrupted, so moving on...")
                        continue
                    print("")

        if NUMPY_SCIPY:
            print("PLATFORM: SciPy")
            try:
                do_benchmark(setup_spgemm_scipy, mul_numpy, SPGEMM_SIZES,
                             sparsity=SPARSITY)
            except KeyboardInterrupt:
                print(" !!! Interrupted, so moving on...")
            print("")

    #
    # Sparse matrix-vector multiplication
    #

    elif arg == "spgemv":

        print("OPERATION: Sparse (%f) Matrix-Vector Product" % SPARSITY)

        if PYVIENNACL:
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    print("PLATFORM: PyViennaCL: %s" % device.name)
                    try:
                        do_benchmark(setup_spgemv_pyvcl, mul_pyvcl, SPGEMV_SIZES,
                                     sparsity=SPARSITY, cl_device=device)
                    except KeyboardInterrupt:
                        print(" !!! Interrupted, so moving on...")
                        continue
                    print("")

        if NUMPY_SCIPY:
            print("PLATFORM: NumPy")
            try:
                do_benchmark(setup_spgemv_scipy, mul_numpy, SPGEMV_SIZES,
                             sparsity=SPARSITY)
            except KeyboardInterrupt:
                print(" !!! Interrupted, so moving on...")
            print("")

    else:

        print("Choose a benchmark: add, gemv, gemm, spgemv, or spgemm")

        # TODO: sparse-vec mult (theano)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        #print("-------------------------------------------------------------")
        #if ADD: main("add")
        #print("-------------------------------------------------------------")
        #if GEMV: main("gemv")
        #print("-------------------------------------------------------------")
        #if GEMM: main("gemm")
        #print("-------------------------------------------------------------")
        #if SPGEMV: main("spgemv")
        #print("-------------------------------------------------------------")
        #if SPGEMM: main("spgemm")
        #print("-------------------------------------------------------------")
        main("") # Because Python leaks memory otherwise ...

