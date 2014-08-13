#!/usr/bin/env python3

from __future__ import division
import numpy as np

#
# Default configuration options
#

# Times for which to run each benchmark loop

TEST_TIME = 0.5 # seconds
QUICK_TEST_TIME = 0.2 # seconds

# Data types to test

DTYPES = ['float32', 'float64'] # TODO: integers

# Benchmarks to run by default

BENCHMARKS = ['add', 'gemv', 'gemm']#,
              #'spgemv', 'spgemm']

# Matrix structure parameters

ADD_SIZES = [int(10**(x/3)) for x in range(12,27)]
GEMV_SIZES = [int(10**(x/5)) for x in range(8,22)]
GEMM_SIZES = [int(10**(x/5)) for x in range(8,21)]
SPGEMV_SIZES = [int(10**(x/5)) for x in range(8,22)]
SPGEMM_SIZES = [int(10**(x/5)) for x in range(8,21)]

QUICK_ADD_SIZE = [10**7]
QUICK_GEMV_SIZE = [3500]
QUICK_GEMM_SIZE = [2100]
QUICK_SPGEMV_SIZE = [3500]
QUICK_SPGEMM_SIZE = [2100]

SPARSITY = 0.02

################################################################################

import argparse, itertools, sys, time
from pyviennacl import Statement
from pyviennacl.backend import Context
from timeit import timeit
try: import gnumpy
except ImportError: CUDA = False
try: import pyopencl as cl
except ImportError: PYVIENNACL = False

class UnsupportedPlatformException(Exception): pass

def get_cl_devices():
    return [x for platform in cl.get_platforms() for x in platform.get_devices()]

def setup_add_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    WITH_PYVCL = True
    try:
        import pyviennacl as p
        import pyopencl as cl
    except: WITH_PYVCL = False
    if not WITH_PYVCL:
        raise UnsupportedPlatformException("PyViennaCL")

    x = p.Vector(size, dtype(1.0), dtype=dtype, context=context)
    y = p.Vector(size, dtype(1.0), dtype=dtype, context=context)

    return x, y

def setup_gemm_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    WITH_PYVCL = True
    try:
        import pyviennacl as p
        import pyopencl as cl
    except: WITH_PYVCL = False
    if not WITH_PYVCL:
        raise UnsupportedPlatformException("PyViennaCL")

    A = p.Matrix(size, size, dtype(1.0), dtype=dtype, context=context)
    B = p.Matrix(size, size, dtype(1.0), dtype=dtype, context=context)

    return A, B

def setup_gemv_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    WITH_PYVCL = True
    try:
        import pyviennacl as p
        import pyopencl as cl
    except: WITH_PYVCL = False
    if not WITH_PYVCL:
        raise UnsupportedPlatformException("PyViennaCL")

    A = p.Matrix(size, size, dtype(1.0), dtype=dtype, context=context)
    x = p.Vector(size, dtype(1.0), dtype=dtype, context=context)

    return A, x

def setup_spgemv_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    WITH_PYVCL = True
    try:
        import pyviennacl as p
        import pyopencl as cl
    except: WITH_PYVCL = False
    if not WITH_PYVCL:
        raise UnsupportedPlatformException("PyViennaCL")
    import math

    nnz = int(max(1, math.ceil((size*size)*sparsity)))
    mod = nnz

    x = p.Vector(size, dtype(1.0), dtype=dtype, context=context)

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

    A = p.CompressedMatrix(data, shape=(size, size), nnz=nnz,
                           dtype=dtype, context=context)
    A.flush()

    return A, x

def setup_spgemm_pyvcl(size, sparsity = None, context = None, dtype = np.float32):
    WITH_PYVCL = True
    try:
        import pyviennacl as p
        import pyopencl as cl
    except: WITH_PYVCL = False
    if not WITH_PYVCL:
        raise UnsupportedPlatformException("PyViennaCL")
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

    A = p.CompressedMatrix(data, shape=(size, size), nnz=nnz,
                           dtype=dtype, context=context)
    A.flush()

    B = p.CompressedMatrix(data, shape=(size, size), nnz=nnz,
                           dtype=dtype, context=context)
    B.flush()

    return A, B

def setup_add_gnumpy(size, sparsity = None, context = None, dtype = np.float32):
    WITH_GNUMPY = True
    try: import gnumpy as gnp
    except: WITH_GNUMPY = False
    if not WITH_GNUMPY:
        raise UnsupportedPlatformException("gnumpy")

    x = np.ones(size).astype(dtype) * 0.3
    y = np.ones(size).astype(dtype) * 0.9

    x = gnp.garray(x)
    y = gnp.garray(y)

    return x, y

def setup_gemm_gnumpy(size, sparsity = None, context = None, dtype = np.float32):
    WITH_GNUMPY = True
    try: import gnumpy as gnp
    except: WITH_GNUMPY = False
    if not WITH_GNUMPY:
        raise UnsupportedPlatformException("gnumpy")

    A = np.ones((size,size)).astype(dtype) * 0.7
    B = np.ones((size,size)).astype(dtype) * 0.5

    A = gnp.garray(A)
    B = gnp.garray(B)

    return A, B

def setup_gemv_gnumpy(size, sparsity = None, context = None, dtype = np.float32):
    WITH_GNUMPY = True
    try: import gnumpy as gnp
    except: WITH_GNUMPY = False
    if not WITH_GNUMPY:
        raise UnsupportedPlatformException("gnumpy")

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
    WITH_SCIPY = True
    try: import scipy.sparse as sp
    except: WITH_SCIPY = False
    if not WITH_SCIPY:
        raise UnsupportedPlatformException("scipy.sparse")
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
    WITH_SCIPY = True
    try: import scipy.sparse as sp
    except: WITH_SCIPY = False
    if not WITH_SCIPY:
        raise UnsupportedPlatformException("scipy.sparse")
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

def gt(x, y): return x > y

def lt(x, y): return x < y

def add_perf(dtype, size, sparsity, time):
    return (np.dtype(dtype).itemsize * 3 * size) / (time*(10**9)), 'GB/s', gt

def gemv_perf(dtype, size, sparsity, time):
    return (np.dtype(dtype).itemsize * (size**2)) / (time * (10**9)), 'GB/s', gt

def gemm_perf(dtype, size, sparsity, time):
    return (2 * (size**3)) / (time * (10**9)), 'GFLOP/s', gt

def spgemv_perf(dtype, size, sparsity, time):
    return size*size*sparsity/time, 'Nonzeros/s', gt

def spgemm_perf(dtype, size, sparsity, time):
    return size*size*sparsity/time, 'Nonzeros/s', gt

benchmarks = {
    'add' : {
        'DESCRIPTION' : 'Vector Addition x = y + z',
        'CONFIG' : (ADD_SIZES, QUICK_ADD_SIZE, None, add_perf),
        'pyviennacl' : (setup_add_pyvcl, add_pyvcl),
        'numpy_scipy' : (setup_add_numpy, add_numpy),
        'gnumpy': (setup_add_gnumpy, add_numpy)
    },
    'gemv' : {
        'DESCRIPTION' : 'Dense Matrix-Vector Product A * x',
        'CONFIG' : (GEMV_SIZES, QUICK_GEMV_SIZE, None, gemv_perf),
        'pyviennacl' : (setup_gemv_pyvcl, gemv_pyvcl),
        'numpy_scipy' : (setup_gemv_numpy, gemv_numpy),
        'gnumpy': (setup_gemv_gnumpy, gemv_numpy)
    },
    'gemm' : {
        'DESCRIPTION' : 'Dense Matrix-Matrix Product A.T * B',
        'CONFIG' : (GEMM_SIZES, QUICK_GEMM_SIZE, None, gemm_perf),
        'pyviennacl' : (setup_gemm_pyvcl, gemm_pyvcl),
        'numpy_scipy' : (setup_gemm_numpy, gemm_numpy),
        'gnumpy': (setup_gemm_gnumpy, gemm_numpy)
    },
    'spgemv' : {
        'DESCRIPTION' : 'Sparse Matrix-Vector Product',
        'CONFIG' : (SPGEMV_SIZES, QUICK_SPGEMV_SIZE, SPARSITY, spgemv_perf),
        'pyviennacl' : (setup_spgemv_pyvcl, spgemv_pyvcl),
        'numpy_scipy' : (setup_spgemv_scipy, spgemv_numpy),
    },
    'spgemm' : {
        'DESCRIPTION' : 'Sparse Matrix-Matrix Product',
        'CONFIG' : (SPGEMM_SIZES, QUICK_SPGEMM_SIZE, SPARSITY, spgemm_perf),
        'pyviennacl' : (setup_spgemm_pyvcl, spgemm_pyvcl),
        'numpy_scipy' : (setup_spgemm_scipy, spgemm_numpy),
    },
}

platforms = {
    'pyviennacl' : ('PyViennaCL', get_cl_devices),
    'numpy_scipy' : ('NumPy/SciPy', None),
    'gnumpy' : ('gnumpy', None)
}

def do_benchmark(platform_id, benchmark_id, dtype=np.float32, quick=False, skip_cpu=False):
    benchmark = benchmarks[benchmark_id]
    platform = platforms[platform_id]

    setup = benchmark[platform_id][0]
    do_op = benchmark[platform_id][1]

    best_perf, unit, order = {}, None, None

    perf_metric = benchmark['CONFIG'][3]
    sparsity = benchmark['CONFIG'][2]

    if quick:
        test_time = TEST_TIME
        sizes = benchmark['CONFIG'][1]
        if len(sizes) > 1:
            raise Exception("Not quick if more than one size used!")
        def quiet_print(*args): pass
    else:
        test_time = QUICK_TEST_TIME
        sizes = benchmark['CONFIG'][0]
        def quiet_print(*args): print(",".join(args))

    if platform[1] is None:
        devices = [None]
    else:
        devices = platform[1]()

    # Test feasibility
    if devices[0] is None:
        setup(10, 0.1, None, dtype)
    else:
        setup(10, 0.1, Context(cl.Context([devices[0]])), dtype)

    for device in devices:
        try:
            if device is None:
                platform_name = platform[0] + " (" + str(np.dtype(dtype)) + ")"
                ctx = None
            else:
                if skip_cpu and cl.device_type.to_string(device.type) == 'CPU':
                    continue
                platform_name = platform[0] + " on " + device.name + " (" + str(np.dtype(dtype)) + ")"
                ctx = Context(cl.Context([device]))

            best_perf[platform_name] = None
            if not quick: print("PLATFORM: " + platform_name)
        
            for size in sizes:
                try:
                    A, B = setup(size, sparsity=sparsity, context=ctx, dtype=dtype)
                    if ctx is not None: ctx.finish_all_queues()
                    for i in range(3):
                        do_op(A, B, ctx)
            
                    N = 0
                    current_time = 0
                    if ctx is not None: ctx.finish_all_queues() # Just to be sure
                    while current_time < test_time:
                        time_before = time.time()
                        do_op(A, B, ctx)
                        if ctx is not None: ctx.finish_all_queues()
                        current_time += time.time() - time_before
                        N += 1
                    time_per = current_time / N
                    perf, unit, order = perf_metric(dtype, size, sparsity, time_per)
                    if best_perf[platform_name] is None or order(perf, best_perf[platform_name]):
                        best_perf[platform_name] = perf
                    quiet_print("("+",".join(map(str, [size, time_per]))+")")
                except Exception as e:
                    quiet_print("Exception with size %d: %s" % (size, e))
                    break
        except KeyboardInterrupt:
            quiet_print(" !!! Interrupted, so moving on...")
        quiet_print("")
    return best_perf, unit, order


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exhaustive', action='store_true')

    temp_args, temp_unknowns = parser.parse_known_args()
    quick = not temp_args.exhaustive

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exhaustive', action='store_true',
                        help="Run benchmarks over a number of sizes, printing complete results. The default is instead to produce a quick summary.")

    parser.add_argument('--benchmarks', metavar='BENCHMARK', nargs="+", type=str, default=BENCHMARKS,
                        help="Benchmarks to run", choices=benchmarks.keys())

    parser.add_argument('--dtypes', metavar='DTYPE', nargs="+", type=str, default=DTYPES,
                        help="Data type(s) to benchmark")

    platform_names = list(platforms.keys())
    platform_names.sort()
    parser.add_argument('--platforms', metavar='PLATFORM', nargs="+", type=str, default=platform_names,
                        help="Platforms to benchmark", choices=platform_names)

    parser.add_argument('--opencl-skip-cpu', action='store_true',
                        help="Skip CPU devices when using PyViennaCL's OpenCL backend")

    for benchmark in benchmarks.keys():
        if quick:
            sizes = benchmarks[benchmark]['CONFIG'][1]
        else:
            sizes = benchmarks[benchmark]['CONFIG'][0]
        parser.add_argument('--' + benchmark + '_sizes', metavar='SIZE', nargs="+", type=int, default=sizes,
                            help="Sizes to benchmark for the " + benchmark + " benchmark")
        sparsity = benchmarks[benchmark]['CONFIG'][2]
        if  sparsity is not None:
            parser.add_argument('--' + benchmark + '_sparsity', metavar='SPARSITY', type=int, default=sparsity,
                                help="Sparsity for the matrices in the " + benchmark + " benchmark")

    args = parser.parse_args()

    results = {}

    previous_benchmark = None
    for benchmark, dtype, platform in itertools.product(args.benchmarks, args.dtypes, args.platforms):
        if platform not in benchmarks[benchmark].keys():
            continue

        if not quick and benchmark != previous_benchmark:
            if previous_benchmark is not None:
                print('')
            print("OPERATION: " + benchmarks[benchmark]['DESCRIPTION'])
            previous_benchmark = benchmark
        elif quick:
            sys.stdout.write(".")
            sys.stdout.flush()

        sizes = getattr(args, benchmark + "_sizes")
        try:
            sparsity = getattr(args, benchmark + "_sparsity")
        except AttributeError:
            sparsity = None
        perf = benchmarks[benchmark]['CONFIG'][3]
        CONFIG = (sizes, sizes, sparsity, perf)
        benchmarks[benchmark]['CONFIG'] = CONFIG
        dtype = getattr(np, dtype)

        try:
            tmp_results = do_benchmark(platform, benchmark, dtype, quick, args.opencl_skip_cpu)
        except UnsupportedPlatformException:
            continue

        if benchmark not in results.keys():
            results[benchmark] = {}
        if dtype not in results[benchmark].keys():
            results[benchmark][dtype] = {}
        results[benchmark][dtype][platform] = tmp_results

    if quick: print('\n')
    print("Peak-performance results")
    print("========================\n")

    # Find longest platform name
    longest = 0
    for w in results.keys():
        for x in results[w].keys():
            for y in results[w][x].keys():
                for z in results[w][x][y][0].keys():
                    l = len(z)
                    if l > longest: longest = l

    benchmark_keys = list(results.keys())
    benchmark_keys.sort()
    for benchmark in benchmark_keys:
        title = benchmarks[benchmark]['DESCRIPTION']
        print(title)
        print('-' * len(title))
        for dt in results[benchmark].keys():
            plat_keys = list(results[benchmark][dt].keys())
            plat_keys.sort()
            for plat in plat_keys:
                for platform_name in results[benchmark][dt][plat][0].keys():
                    result = results[benchmark][dt][plat][0][platform_name]
                    if not result: continue
                    if len(platform_name) < longest:
                        whitespace = " " * (longest - len(platform_name))
                    else:
                        whitespace = ""
                    print(platform_name + ":" + whitespace + " %.2f" % result + " " + results[benchmark][dt][plat][1])
        print('')


if __name__ == "__main__": main()
