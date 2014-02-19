#!/usr/bin/env python

import math
import numpy as np
import os
import pyviennacl as p
import random
import sys


def test_scalar(epsilon, dtype):
    """
    Basic arithmetic:
    + add, sub, mul, div
    
    Scalar result types:
    
    Host and Device scalars
    """
    a = dtype(random.random())
    b = dtype(random.random())
    c = dtype(random.random())
    d = dtype(random.random())

    alpha = p.Scalar(a)
    beta = p.Scalar(b)
    gamma = p.HostScalar(c)
    delta = p.HostScalar(d)

    if not alpha == a:
        raise RuntimeError("Failed to initialise device scalar")
    if not beta == b:
        raise RuntimeError("Failed to initialise device scalar")
    if not gamma == c:
        raise RuntimeError("Failed to initialise host scalar")
    if not delta == d:
        raise RuntimeError("Failed to initialise host scalar")
    print("Test: scalar initialisation passed")

    ### Test copy
    A = alpha.copy()
    if A != a:
        raise RuntimeError("Failed to copy device scalar")
    G = gamma.copy()
    if G != c:
        raise RuntimeError("Failed to copy host scalar")
    print("Test: scalar copy passed")
    
    ### Test inter-type initialisation
    A_tmp = A
    A = p.Scalar(G)
    if A != c:
        raise RuntimeError("Failed to initialise device scalar from copied host scalar")
    B = p.HostScalar(beta)
    if B != b:
        raise RuntimeError("Failed to initialise host scalar from device scalar")
    G = p.HostScalar(A_tmp)
    if G != a:
        raise RuntimeError("Failed to initialise host scalar from copied device scalar")
    D = p.Scalar(delta)
    if D != d:
        raise RuntimeError("Failed to initialise device scalar from host scalar")
    print("Test: inter-type scalar initialisation passed")

    ### pyvcl type arithmetic
    X = (a / c) + (b ** a) * (c - d) // b
    Y = (alpha / gamma) + (beta ** alpha) * (gamma - delta) // beta
    X += a
    Y += alpha
    X -= b
    Y -= beta
    X *= c
    Y *= gamma
    X /= d
    Y /= delta
    X **= dtype(2)
    Y **= p.HostScalar(dtype(2))
    X //= Y
    Y //= Y
    if (X - Y) > epsilon:
        raise RuntimeError("Failed basic arithmetic test")
    print("Test: basic arithmetic passed")

    ### Inter-type arithmetic
    X = (a / gamma) + (b ** alpha) * (c - delta) // beta
    Y = (alpha / c) + (beta ** a) * (gamma - d) // b
    X = p.Scalar(X, dtype = dtype)
    X += alpha
    Y += a
    X -= b
    Y -= beta
    X *= gamma
    Y *= c
    X /= d
    Y /= delta
    X **= p.HostScalar(dtype(2))
    Y **= dtype(2)
    if (X - Y) > epsilon:
        raise RuntimeError("Failed inter-type arithmetic test")
    print("Test: inter-type arithmetic passed")

    ### Scalar result type arithmetic
    """
    + Norm_1, Norm_2, Norm_Inf
    + Element* operations?
    + Dot (ie, inner product)
    """
    vec = p.Vector([X, a, beta, c, delta, Y], dtype = dtype)
    r1 = vec.norm(1)
    r2 = vec.norm(2)
    r3 = vec.norm(p.inf)
    r4 = vec.dot(vec * alpha)
    R1 = r1.value
    R2 = r2.value
    R3 = r3.value
    R4 = r4.value
    X = (r1 * a + r2 * beta - r3 / c - r4 // delta) * (r1 + r4 - R3)
    Y = (R1 * a + R2 * beta - R3 / c - R4 // delta) * (R1 + R4 - r3)
    if (X - Y) > epsilon:
        raise RuntimeError("Failed scalar result type arithmetic test")
    print("Test: scalar result type arithmetic passed")


def test():
    print("----------------------------------------------")
    print("----------------------------------------------")
    print("## Test :: Scalar operations")
    print("----------------------------------------------")
    print("----------------------------------------------")
    print()
    print("----------------------------------------------")

    print("*** Using float numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-3
    print("  eps:      %s" % epsilon)
    test_scalar(epsilon, p.float32)

    print("*** Using double numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-11
    print("  eps:      %s" % epsilon)
    test_scalar(epsilon, p.float64)

    print("# Test passed")
    
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(test())


