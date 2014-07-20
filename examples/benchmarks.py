#!/usr/bin/env python3

from __future__ import division

#
# Default configuration options
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
GEMV_SIZES = [int(10**(x/5)) for x in range(8,22)]
GEMM_SIZES = [int(10**(x/5)) for x in range(8,21)]
SPGEMV_SIZES = [int(10**(x/5)) for x in range(8,22)]
SPGEMM_SIZES = [int(10**(x/5)) for x in range(8,21)]

QUICK_ADD_SIZES = [int(10**(x/3)) for x in range(20,22)]
QUICK_GEMV_SIZES = [int(10**(x/5)) for x in range(18,21)]
QUICK_GEMM_SIZES = [int(10**(x/5)) for x in range(18,21)]
QUICK_SPGEMV_SIZES = [int(10**(x/5)) for x in range(18,21)]
QUICK_SPGEMM_SIZES = [int(10**(x/5)) for x in range(18,21)]

SPARSITY = 0.02

################################################################################

import argparse, sys, time
import numpy as np
from pyviennacl import Statement
from pyviennacl.backend import Context
from timeit import timeit
try: import gnumpy
except ImportError: CUDA = False
try: import pyopencl as cl
except ImportError: PYVIENNACL = False

def get_cl_devices():
    return [x for platform in cl.get_platforms() for x in platform.get_devices()]

def setup_add_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    import pyviennacl as p

    x = p.Vector(size, 1.0, dtype=dtype, context=context)
    y = p.Vector(size, 1.0, dtype=dtype, context=context)

    return x, y

def setup_gemm_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    import pyviennacl as p

    A = p.Matrix(size, size, 1.0, dtype=dtype, context=context)
    B = p.Matrix(size, size, 1.0, dtype=dtype, context=context)

    return A, B

def setup_gemv_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    import pyviennacl as p

    A = p.Matrix(size, size, 1.0, dtype=dtype, context=context)
    x = p.Vector(size, 1.0, dtype=dtype, context=context)

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

def setup_add_gnumpy(size, sparsity = None, context = None, dtype = np.float32):
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

def setup_add_numpy(size, sparsity = None, context = None, dtype = np.float32):
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
    return Statement(A+B).execute()
    #return (A+B).execute()

def gemv_pyvcl(A, x, ctx = None):
    return Statement(A*x).execute()
    #return (A*x).execute()

def gemm_pyvcl(A, B, ctx = None):
    return Statement(A.T*B).execute()
    #return (A.T*B).execute()

def spgemv_pyvcl(A, x, ctx = None):
    return Statement(A*x).execute()
    #return (A*x).execute()

def spgemm_pyvcl(A, B, ctx = None):
    return Statement(A*B).execute()
    #return (A*B).execute()

def add_numpy(A, B, ctx = None):
    return A+B

def gemv_numpy(A, x, ctx = None):
    return A.dot(x)

def gemm_numpy(A, B, ctx = None):
    return A.T.dot(B)

def spgemv_numpy(A, x, ctx = None):
    return A.dot(x)

def spgemm_numpy(A, B, ctx = None):
    return A.dot(B)

# TODO general parameters (sizes, sparsities, etc)
benchmarks = {
    'add' : {
        'DESCRIPTION' : 'Vector Addition x = y + z',
        'CONFIG' : (ADD_SIZES, QUICK_ADD_SIZES, None),
        'pyviennacl' : (setup_add_pyvcl, add_pyvcl),
        'numpy_scipy' : (setup_add_numpy, add_numpy),
        'gnumpy': (setup_add_gnumpy, add_numpy)
    },
    'gemv' : {
        'DESCRIPTION' : 'Dense Matrix-Matrix Product A.T * B',
        'CONFIG' : (GEMV_SIZES, QUICK_GEMV_SIZES, None),
        'pyviennacl' : (setup_gemv_pyvcl, gemv_pyvcl),
        'numpy_scipy' : (setup_gemv_numpy, gemv_numpy),
        'gnumpy': (setup_gemv_gnumpy, gemv_numpy)
    },
    'gemm' : {
        'DESCRIPTION' : 'Dense Matrix-Vector Product A * x',
        'CONFIG' : (GEMM_SIZES, QUICK_GEMM_SIZES, None),
        'pyviennacl' : (setup_gemm_pyvcl, gemm_pyvcl),
        'numpy_scipy' : (setup_gemm_numpy, gemm_numpy),
        'gnumpy': (setup_gemm_gnumpy, gemm_numpy)
    },
    'spgemv' : {
        'DESCRIPTION' : 'Sparse (%f) Matrix-Vector Product',
        'CONFIG' : (SPGEMV_SIZES, QUICK_SPGEMV_SIZES, SPARSITY),
        'pyviennacl' : (setup_spgemv_pyvcl, spgemv_pyvcl),
        'numpy_scipy' : (setup_spgemv_scipy, spgemv_numpy),
    },
    'spgemm' : {
        'DESCRIPTION' : 'Sparse (%f) Matrix-Matrix Product',
        'CONFIG' : (SPGEMM_SIZES, QUICK_SPGEMM_SIZES, SPARSITY),
        'pyviennacl' : (setup_spgemm_pyvcl, spgemm_pyvcl),
        'numpy_scipy' : (setup_spgemm_scipy, spgemm_numpy),
    },
}

platforms = {
    'pyviennacl' : ('PyViennaCL', get_cl_devices),
    'numpy_scipy' : ('NumPy/SciPy', None),
    'gnumpy' : ('gnumpy', None)
}

def do_benchmark(platform_id, benchmark_id, quick=False):
    benchmark = benchmarks[benchmark_id]
    platform = platforms[platform_id]

    setup = benchmark[platform_id][0]
    do_op = benchmark[platform_id][1]

    sparsity = benchmark['CONFIG'][2]

    if quick:
        sizes = benchmark['CONFIG'][1]
    else:
        sizes = benchmark['CONFIG'][0]

    print("OPERATION: " + benchmark['DESCRIPTION'])

    if platform[1] is None:
        devices = [None]
    else:
        devices = platform[1]()

    for device in devices:
        try:
            if device is None:
                print("PLATFORM: " + platform[0])
                ctx = None
            else:
                print("PLATFORM: " + platform[0] + ": " + device.name)
                ctx = Context(cl.Context([device]))
        
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
                        if ctx is not None: ctx.finish_all_queues()
                        current_time += time.time() - time_before
                        N += 1
                    print(size, current_time / N)
                except Exception as e:
                    print("Exception with size %d: %s" % (size, e))
                    break
        except KeyboardInterrupt:
            print(" !!! Interrupted, so moving on...")
        print("")


def main(args):
    do_benchmark('pyviennacl', 'gemm', True)


if __name__ == "__main__":
    main(sys.argv)
