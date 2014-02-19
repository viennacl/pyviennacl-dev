#!/usr/bin/env python

import math
import os
import pyviennacl as p
import sys

from test_common import diff, test_matrix_layout, test_matrix_slice


def run_test(*args, **kwargs):
    """
    A, A_trans, B, B_trans must be numpy array or matrix instances
    """
    epsilon = args[0]
    A = args[1]
    A_trans = args[2]
    B = args[3]
    B_trans = args[4]
    C = args[5]
    vcl_A = args[6]
    vcl_A_trans = args[7]
    vcl_B = args[8]
    vcl_B_trans = args[9]
    vcl_C = args[10]

    act_diff = math.fabs(diff(A, vcl_A))
    if act_diff > epsilon:
        raise Exception("Error copying A")

    act_diff = math.fabs(diff(B, vcl_B))
    if act_diff > epsilon:
        x = p.Matrix(vcl_B.shape, dtype = vcl_B.dtype, layout = p.ROW_MAJOR)
        p.Assign(x, vcl_B).execute()
        print(x.value)
        print(B)
        print(x == B)
        print(act_diff)
        raise Exception("Error copying B")

    #act_diff = math.fabs(diff(C, vcl_C))
    #if act_diff > epsilon:
    #    raise Exception("Error copying C")

    act_diff = math.fabs(diff(A_trans, vcl_A_trans))
    if act_diff > epsilon:
        raise Exception("Error copying A_trans")

    act_diff = math.fabs(diff(B_trans, vcl_B_trans))
    if act_diff > epsilon:
        raise Exception("Error copying B_trans")

    #act_diff = math.fabs(diff(C_trans, vcl_C_trans))
    #if act_diff > epsilon:
    #    raise Exception("Error copying C_trans")

    #A = vcl_A.value
    #A_trans = vcl_A_trans.value
    #B = vcl_B.value
    #B_trans = vcl_B_trans.value
    #C = vcl_C.value
    #C_trans = C.T

    # C +-= A * B
    C = A.dot(B)
    vcl_C = vcl_A * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C = A * B passed!")

    C += A.dot(B)
    vcl_C += vcl_A * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C += A * B passed!")

    C -= A.dot(B)
    vcl_C -= vcl_A * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C -= A * B passed!")

    # C +-= A * trans(B)
    C = A.dot(B_trans.T)
    vcl_C = vcl_A * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C = A * trans(B) passed!")

    C += A.dot(B_trans.T)
    vcl_C += vcl_A * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C += A * trans(B) passed!")

    C -= A.dot(B_trans.T)
    vcl_C -= vcl_A * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C -= A * trans(B) passed!")

    # C +-= trans(A) * B
    C = A_trans.T.dot(B)
    vcl_C = vcl_A_trans.T * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C = trans(A) * B passed!")

    C += A_trans.T.dot(B)
    vcl_C += vcl_A_trans.T * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C += trans(A) * B passed!")

    C -= A_trans.T.dot(B)
    vcl_C -= vcl_A_trans.T * vcl_B
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C -= trans(A) * B passed!")

    # C +-= trans(A) * trans(B)
    C = A_trans.T.dot(B_trans.T)
    vcl_C = vcl_A_trans.T * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C = trans(A) * trans(B) passed!")

    C += A_trans.T.dot(B_trans.T)
    vcl_C += vcl_A_trans.T * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C += trans(A) * trans(B) passed!")

    C -= A_trans.T.dot(B_trans.T)
    vcl_C -= vcl_A_trans.T * vcl_B_trans.T
    act_diff = math.fabs(diff(C, vcl_C))
    if (act_diff > epsilon):
        raise Exception("Error at operation: matrix-matrix product; diff = %s"
                        % act_diff)
    print("Test C -= trans(A) * trans(B) passed!")

    return os.EX_OK


def test():
    print("----------------------------------------------")
    print("----------------------------------------------")
    print("## Test :: BLAS 3 routines")
    print("----------------------------------------------")
    print("----------------------------------------------")
    print()
    print("----------------------------------------------")
    print("--- Part 1: Testing matrix-matrix products ---")

    print("*** Using float numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-3
    print("  eps:      %s" % epsilon)
    test_matrix_layout(test_matrix_slice, run_test,
                       epsilon, p.float32) #, 11, 11, 11)

    print("*** Using double numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-11
    print("  eps:      %s" % epsilon)
    test_matrix_layout(test_matrix_slice, run_test,
                       epsilon, p.float64)

    print("# Test passed")
    
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(test())

