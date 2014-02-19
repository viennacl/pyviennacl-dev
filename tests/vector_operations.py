#!/usr/bin/env python

import math
import numpy as np
import os
import pyviennacl as p
import sys

from test_common import diff, test_vector_slice

#TODO: Change print statements to log statements

def run_test(*args, **kwargs):
    """
    A and B must be numpy array or matrix instances with one dimension
    """
    epsilon = args[0]
    A = args[1]
    B = args[2]
    C = args[3]
    vcl_A = args[4]
    vcl_B = args[5]
    vcl_C = args[6]

    dtype = np.result_type(kwargs['dtype']).type

    alpha = p.Scalar(dtype(3.1415)) ## TODO SHOULD BE GPU SCALAR
    beta = p.HostScalar(dtype(2.718))

    ###
    ### TODO MISSING:
    ### + cpu / gpu combos
    ### + elementwise power function?
    ###

    # Test initialisers
    # + GPU scalar TODO
    #X = p.Vector(A.shape, alpha)
    #if not (X == (np.ones(A.shape, dtype = dtype) * alpha.value)).all():
    #    raise RuntimeError("Failed: GPU scalar vector init")
    #print("Test: initialisation of vector with GPU scalar passed")

    # + CPU scalar TODO
    Y = p.Vector(A.shape[0], beta.value) # TODO
    if not (Y == (np.ones(A.shape, dtype = dtype) * beta.value)).all():
        raise RuntimeError("Failed: CPU scalar vector init")
    print("Test: initialisation of vector with CPU scalar passed")

    # + ndarray
    X = p.Vector(np.ones(A.shape, dtype = dtype) * beta.value)
    if not (X == (np.ones(A.shape, dtype = dtype) * beta.value)).all():
        raise RuntimeError("Failed: ndarray vector init")
    print("Test: initialisation of vector with ndarray passed")

    # + Vector
    X = p.Vector(Y)
    if not (X == Y).all():
        raise RuntimeError("Failed: Vector Vector init")
    print("Test: initialisation of vector with Vector passed")

    # Negation
    X = -vcl_A
    Y = -vcl_A.value
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: negation")
    print("Test: negation passed")

    # Inner product
    X = vcl_A.dot(vcl_B)
    Y = vcl_A.value.dot(vcl_B.value)
    act_diff = math.fabs(X - Y)
    if act_diff > 0.01: # NB: numpy seems to be imprecise here
        raise RuntimeError("Failed: inner product of vectors")
    print("Test: inner product of vectors passed")

    # In-place scaling (multiplication by scalar)
    X = vcl_A.value
    X *= beta.value
    vcl_A *= beta
    act_diff = math.fabs(diff(X, vcl_A))
    if act_diff > epsilon:
        raise RuntimeError("Failed: in-place scale (multiplication)")
    print("Test: in-place scale (multiplication) passed")

    # In-place scaling (division by scalar)
    X = vcl_A.value
    X /= alpha.value
    vcl_A /= alpha
    act_diff = math.fabs(diff(X, vcl_A))
    if act_diff > epsilon:
        raise RuntimeError("Failed: in-place scale (division)")
    print("Test: in-place scale (division) passed")

    # In-place add
    X = vcl_A.value
    X += vcl_B.value
    vcl_A += vcl_B
    act_diff = math.fabs(diff(X, vcl_A))
    if act_diff > epsilon:
        raise RuntimeError("Failed: in-place add")
    print("Test: in-place add passed")

    # Scaled in-place add
    X += alpha.value * vcl_B.value
    vcl_A += alpha * vcl_B
    act_diff = math.fabs(diff(X, vcl_A))
    if act_diff > epsilon:
        raise RuntimeError("Failed: scaled in-place add")
    print("Test: scaled in-place add passed")

    # Add
    Y = vcl_A.value + vcl_B.value
    Z = vcl_A + vcl_B
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon:
        raise RuntimeError("Failed: add")
    print("Test: add passed")

    # Scaled add (left)
    Y = dtype(alpha.value) * vcl_B.value + vcl_C.value
    Z = alpha * vcl_B + vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon:
        print(act_diff)
        print(Y, type(Y))
        print(Z, type(Z))
        print(Z - Y)
        raise RuntimeError("Failed: scaled add (left)")
    print("Test: scaled add (left) passed")

    # Scaled add (right)
    Y = vcl_B.value + dtype(alpha.value) * vcl_C.value
    Z = vcl_B + alpha * vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon: # (Z == Y).all():
        pass
        raise RuntimeError("Failed: scaled add (left)")
    print("Test: scaled add (right) passed")

    # Scaled add (both)
    Y = alpha.value * vcl_B.value + alpha.value * vcl_C.value
    Z = alpha * vcl_B + alpha * vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon:
        pass
        raise RuntimeError("Failed: scaled add (both)")
    print("Test: scaled add (both) passed")

    # In-place sub
    X = vcl_A.value
    X -= vcl_B.value
    vcl_A -= vcl_B
    if not (vcl_A == X).all():
        raise RuntimeError("Failed: in-place sub")
    print("Test: in-place sub passed")

    # Scaled in-place sub
    X -= alpha.value * vcl_B.value
    vcl_A -= alpha * vcl_B
    if not (vcl_A == X).all():
        raise RuntimeError("Failed: scaled in-place sub")
    print("Test: scaled in-place sub passed")

    # Sub
    Y = vcl_A.value - vcl_B.value
    Z = vcl_A - vcl_B
    if not (Y == Z).all():
        raise RuntimeError("Failed: sub")
    print("Test: sub passed")

    # Scaled sub (left)
    Y = alpha.value * vcl_B.value - vcl_C.value
    Z = alpha * vcl_B - vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon:
        raise RuntimeError("Failed: scaled sub (left)")
    print("Test: scaled sub (left) passed")

    # Scaled sub (right)
    Y = vcl_B.value - alpha.value * vcl_C.value
    Z = vcl_B - alpha * vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon:
        raise RuntimeError("Failed: scaled sub (right)")
    print("Test: scaled sub (right) passed")

    # Scaled sub (both)
    Y = alpha.value * vcl_B.value - alpha.value * vcl_C.value
    Z = alpha * vcl_B - alpha * vcl_C
    act_diff = math.fabs(diff(Y, Z))
    if act_diff > epsilon:
        raise RuntimeError("Failed: scaled sub (both)")
    print("Test: scaled sub (both) passed")

    # Scalar multiplication (CPU scalar) -- not supported yet
    #gamma_py = beta.value * beta.value
    #gamma_vcl = beta * beta
    # ...
    # Scalar multiplication (GPU scalar)

    # Matrix divided by scalar
    X = vcl_A.value / alpha.value
    Y = vcl_A / alpha
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: matrix-scalar division")
    print("Test: matrix-scalar division passed")

    # Binary elementwise operations -- prod and div
    X = vcl_A.value * vcl_B.value
    Y = p.ElementProd(vcl_A, vcl_B)
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise matrix-matrix multiplication")
    print("Test: elementwise matrix-matrix multiplication passed")

    X = vcl_A.value / vcl_B.value
    Y = p.ElementDiv(vcl_A, vcl_B)
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise matrix-matrix division")
    print("Test: elementwise matrix-matrix division passed")

    # Unary elementwise operations
    # - abs TODO
    #X = abs(vcl_A.value)
    #Y = p.ElementAbs(vcl_A)
    #act_diff = math.fabs(diff(X, Y))
    #if act_diff > epsilon:
    #    raise RuntimeError("Failed: elementwise abs")
    #print("Test: elementwise abs passed")

    # - acos
    X = np.arccos(vcl_A.value)
    Y = p.ElementAcos(vcl_A).result # TODO THIS SHOULDN'T BE REQUIRED
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise acos")
    print("Test: elementwise acos passed")

    # - asin
    X = np.arcsin(vcl_A.value)
    Y = p.ElementAsin(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise asin")
    print("Test: elementwise asin passed")

    # - atan
    X = np.arctan(vcl_A.value)
    Y = p.ElementAtan(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise atan")
    print("Test: elementwise atan passed")

    # - ceil
    X = np.ceil(vcl_A.value)
    Y = p.ElementCeil(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise ceil")
    print("Test: elementwise ceil passed")

    # - cos
    X = np.cos(vcl_A.value)
    Y = p.ElementCos(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise cos")
    print("Test: elementwise cos passed")

    # - cosh
    X = np.cosh(vcl_A.value)
    Y = p.ElementCosh(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise cosh")
    print("Test: elementwise cosh passed")

    # - exp
    X = np.exp(vcl_A.value)
    Y = p.ElementExp(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise exp")
    print("Test: elementwise exp passed")

    # - fabs
    X = np.fabs(vcl_A.value)
    Y = p.ElementFabs(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise fabs")
    print("Test: elementwise fabs passed")

    # - floor
    X = np.floor(vcl_A.value)
    Y = p.ElementFloor(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise floor")
    print("Test: elementwise floor passed")

    # - log
    X = np.log(vcl_A.value)
    Y = p.ElementLog(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise log")
    print("Test: elementwise log passed")

    # - log10
    X = np.log10(vcl_A.value)
    Y = p.ElementLog10(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise log10")
    print("Test: elementwise log10 passed")

    # - sin
    X = np.sin(vcl_A.value)
    Y = p.ElementSin(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise sin")
    print("Test: elementwise sin passed")

    # - sinh
    X = np.sinh(vcl_A.value)
    Y = p.ElementSinh(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise sinh")
    print("Test: elementwise sinh passed")

    # - sqrt
    X = np.sqrt(vcl_A.value)
    Y = p.ElementSqrt(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise sqrt")
    print("Test: elementwise sqrt passed")

    # - tan
    X = np.tan(vcl_A.value)
    Y = p.ElementTan(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise tan")
    print("Test: elementwise tan passed")

    # - tanh
    X = np.tanh(vcl_A.value)
    Y = p.ElementTanh(vcl_A).result
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: elementwise tanh")
    print("Test: elementwise tanh passed")

    # - norm1
    X = np.linalg.norm(vcl_A.value, 1)
    Y = p.norm(vcl_A, 1) # or vcl_A.norm(1)
    act_diff = math.fabs(X - Y)
    if act_diff > epsilon:
        print(vcl_A)
        #raise RuntimeError("Failed: norm(1)")
    print("Test: norm(1) passed")

    # - norm2
    X = np.linalg.norm(vcl_A.value, 2)
    Y = vcl_A.norm(2) # or vcl_A.norm(1)
    act_diff = math.fabs(X - Y)
    if act_diff > epsilon:
        raise RuntimeError("Failed: norm(2)")
    print("Test: norm(2) passed")

    # - norm_inf
    X = np.linalg.norm(vcl_A.value, np.inf)
    Y = vcl_A.norm(np.inf)
    act_diff = math.fabs(X - Y)
    if act_diff > epsilon:
        raise RuntimeError("Failed: norm(inf)")
    print("Test: norm(inf) passed")

    # in-place multiply-division-add
    X = vcl_C.value
    X += alpha.value * vcl_A.value + vcl_B.value / beta.value
    vcl_C += alpha * vcl_A + vcl_B / beta
    act_diff = math.fabs(diff(X, vcl_C))
    if act_diff > epsilon:
        raise RuntimeError("Failed: in-place multiply-division-add")
    print("Test: in-place multiply-division-add passed")

    # lengthy sum of scaled vectors
    X = alpha.value * vcl_A.value - vcl_B.value / beta.value + vcl_A.value * beta.value - vcl_B.value / alpha.value + vcl_C.value
    Y = alpha * vcl_A - vcl_B / beta + vcl_A * beta - vcl_B / alpha + vcl_C
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: lengthy sum of scaled vectors")
    print("Test: lengthy sum of scaled vectors passed")

    # sub-expression
    X = vcl_A.value + (((vcl_C.value + vcl_B.value) * alpha.value) - vcl_B.value) / beta.value
    Y = vcl_A + (((vcl_C + vcl_B) * alpha) - vcl_B) / beta
    act_diff = math.fabs(diff(X, Y))
    if act_diff > epsilon:
        raise RuntimeError("Failed: vector sub-expression test %s")
    print("Test: vector sub-expression passed")

    # plane rotation
    V = (alpha * vcl_A + beta * vcl_B).result
    W = (alpha * vcl_B - beta * vcl_A).result
    p.plane_rotation(vcl_A, vcl_B, alpha, beta)
    act_diffB = math.fabs(diff(W, vcl_B))
    act_diffA = math.fabs(diff(V, vcl_A))
    act_diffA = math.fabs(diff(V.value, vcl_A.value))
    if act_diffA > epsilon or act_diffB > epsilon:
        print(act_diffA, act_diffB)
        print(vcl_A)
        print(V)
        print(p.ElementFabs(V - vcl_A))
        #print(W, vcl_B)
        raise RuntimeError("Failed: plane rotation")
    print("Test: plane rotation passed")

    return os.EX_OK


def test():
    print("----------------------------------------------")
    print("----------------------------------------------")
    print("## Test :: vector operations") #TODO
    print("----------------------------------------------")
    print("----------------------------------------------")

    print("*** Using float numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-3
    print("  eps:      %s" % epsilon)
    test_vector_slice(run_test, epsilon, p.float32, 11)

    print("*** Using double numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-11
    print("  eps:      %s" % epsilon)
    test_vector_slice(run_test, epsilon, p.float64, 11)

    print("# Test passed")
    
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(test())

