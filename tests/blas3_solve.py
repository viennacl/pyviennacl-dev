#!/usr/bin/env python

import math
import os
import pyviennacl as p
import scipy.linalg as sp
import scipy.sparse.linalg as spsp
import sys

from test_common import diff, test_matrix_layout, test_matrix_solvers

def test_kernel(*args, **kwargs):
    """
    A, A_trans, B, B_trans must be numpy array or matrix instances
    """

    epsilon = args[0]
    A_upper, A_unit_upper, A_lower, A_unit_lower, A_trans_upper, A_trans_unit_upper, A_trans_lower, A_trans_unit_lower = args[1]
    B, B_trans = args[2]
    vcl_A_upper, vcl_A_unit_upper, vcl_A_lower, vcl_A_unit_lower, vcl_A_trans_upper, vcl_A_trans_unit_upper, vcl_A_trans_lower, vcl_A_trans_unit_lower = args[3]
    vcl_B, vcl_B_trans = args[4]

    Bvec = B[::, 0]
    vcl_Bvec = p.Vector(vcl_B.value[::, 0]) # TODO: get rid of .value
    if not (Bvec == vcl_Bvec).all():
        print(Bvec)
        print(vcl_Bvec)
        raise RuntimeError("Failed creating B vector")

    # A \ B
    vcl_X = p.solve(vcl_A_upper, vcl_B, p.upper_tag())
    X = sp.solve(A_upper, B)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A \ B for upper triangular A: %s" % act_diff)
    print("Test passed: solving A \ B for upper triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_unit_upper, vcl_B, p.unit_upper_tag())
    X = sp.solve(A_unit_upper, B)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A \ B for unit upper triangular A: %s" % act_diff)
    print("Test passed: solving A \ B for unit upper triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_lower, vcl_B, p.lower_tag())
    X = sp.solve(A_lower, B, lower = True)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A \ B for lower triangular A: %s" % act_diff)
    print("Test passed: solving A \ B for lower triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_unit_lower, vcl_B, p.unit_lower_tag())
    X = sp.solve(A_unit_lower, B, lower = True)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A \ B for unit lower triangular A: %s" % act_diff)
    print("Test passed: solving A \ B for unit lower triangular A: %s" % act_diff)

    # A^T \ B
    vcl_X = p.solve(vcl_A_trans_upper, vcl_B, p.upper_tag())
    X = sp.solve(A_trans_upper, B)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A^T \ B for upper triangular A: %s" % act_diff)
    print("Test passed: solving A^T \ B for upper triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_trans_unit_upper, vcl_B, p.unit_upper_tag())
    X = sp.solve(A_trans_unit_upper, B)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A^T \ B for unit upper triangular A: %s" % act_diff)
    print("Test passed: solving A^T \ B for unit upper triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_trans_lower, vcl_B, p.lower_tag())
    X = sp.solve(A_trans_lower, B, lower = True)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A^T \ B for lower triangular A: %s" % act_diff)
    print("Test passed: solving A^T \ B for lower triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_trans_unit_lower, vcl_B, p.unit_lower_tag())
    X = sp.solve(A_trans_unit_lower, B, lower = True)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A^T \ B for unit lower triangular A: %s" % act_diff)
    print("Test passed: solving A^T \ B for unit lower triangular A: %s" % act_diff)

    # A \ B^T
    vcl_X = p.solve(vcl_A_upper, vcl_B_trans, p.upper_tag())
    X = sp.solve(A_upper, B_trans)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A \ B^T for upper triangular A: %s" % act_diff)
    print("Test passed: solving A \ B^T for upper triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_unit_upper, vcl_B_trans, p.unit_upper_tag())
    X = sp.solve(A_unit_upper, B_trans)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A \ B^T for unit upper triangular A: %s" % act_diff)
    print("Test passed: solving A \ B^T for unit upper triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_lower, vcl_B_trans, p.lower_tag())
    X = sp.solve(A_lower, B_trans, lower = True)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A \ B^T for lower triangular A: %s" % act_diff)
    print("Test passed: solving A \ B^T for lower triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_unit_lower, vcl_B_trans, p.unit_lower_tag())
    X = sp.solve(A_unit_lower, B_trans, lower = True)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A \ B^T for unit lower triangular A: %s" % act_diff)
    print("Test passed: solving A \ B^T for unit lower triangular A: %s" % act_diff)

    # A^T \ B^T
    vcl_X = p.solve(vcl_A_trans_upper, vcl_B_trans, p.upper_tag())
    X = sp.solve(A_trans_upper, B_trans)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A^T \ B^T for upper triangular A: %s" % act_diff)
    print("Test passed: solving A^T \ B^T for upper triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_trans_unit_upper, vcl_B_trans, p.unit_upper_tag())
    X = sp.solve(A_trans_unit_upper, B_trans)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A^T \ B^T for unit upper triangular A: %s" % act_diff)
    print("Test passed: solving A^T \ B^T for unit upper triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_trans_lower, vcl_B_trans, p.lower_tag())
    X = sp.solve(A_trans_lower, B_trans, lower = True)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A^T \ B^T for lower triangular A: %s" % act_diff)
    print("Test passed: solving A^T \ B^T for lower triangular A: %s" % act_diff)

    vcl_X = p.solve(vcl_A_trans_unit_lower, vcl_B_trans, p.unit_lower_tag())
    X = sp.solve(A_trans_unit_lower, B_trans, lower = True)
    act_diff = math.fabs(diff(X, vcl_X))
    if act_diff > epsilon:
        raise RuntimeError("Failed solving A^T \ B^T for unit lower triangular A: %s" % act_diff)
    print("Test passed: solving A^T \ B^T for unit lower triangular A: %s" % act_diff)

    # !!! NB: ITERATIVE SOLVERS NOT DEFINED FOR DENSE MATRICES CURRENTLY
    #
    # GMRES
    #vcl_X = p.solve(vcl_A_upper, vcl_Bvec, p.gmres_tag(tolerance=(epsilon/10)))
    #X, info = spsp.gmres(A_upper, Bvec, tol=(epsilon/10))
    #act_diff = math.fabs(diff(X, vcl_X))
    #if act_diff > epsilon:
    #    raise RuntimeError("Failed solving A \ b using GMRES: %s" % act_diff)
    #print("Test passed: solving A \ b using GMRES: %s" % act_diff)
    #
    # CG -- TODO: need a symmetric positive definite matrix for test
    #vcl_X = p.solve(vcl_A_upper, vcl_Bvec, p.cg_tag())
    #X, info = spsp.cg(A_upper, Bvec)
    #act_diff = math.fabs(diff(X, vcl_X))
    #if act_diff > epsilon:
    #    raise RuntimeError("Failed solving A \ b using CG: %s" % act_diff)
    #print("Test passed: solving A \ b using CG: %s" % act_diff)
    #
    # BiCGStab -- TODO: need a non-symmetric matrix for test
    #vcl_X = p.solve(vcl_A_upper, vcl_Bvec, p.cg_tag())
    #X, info = spsp.cg(A_upper, Bvec)
    #act_diff = math.fabs(diff(X, vcl_X))
    #if act_diff > epsilon:
    #    raise RuntimeError("Failed solving A \ b using CG: %s" % act_diff)
    #print("Test passed: solving A \ b using CG: %s" % act_diff)

    # TODO: in-place solvers
    # TODO: iterative solvers on sparse matrices

    return os.EX_OK


def test():
    print("----------------------------------------------")
    print("----------------------------------------------")
    print("## Test :: BLAS 3 routines :: solvers")
    print("----------------------------------------------")
    print("----------------------------------------------")
    print()
    print("----------------------------------------------")

    print("*** Using float numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-1
    print("  eps:      %s" % epsilon)
    test_matrix_layout(test_matrix_solvers, test_kernel, 
                       epsilon, p.float32,
                       9, 9, 9,
                       num_matrices = 2)

    print("*** Using double numeric type ***")
    print("# Testing setup:")
    epsilon = 1.0E-08
    print("  eps:      %s" % epsilon)
    test_matrix_layout(test_matrix_solvers, test_kernel,
                       epsilon, p.float64,
                       9, 9, 9,
                       num_matrices = 2)

    print("# Test passed")
    
    return os.EX_OK


if __name__ == "__main__":
    sys.exit(test())
