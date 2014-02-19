#!/usr/bin/env python

import math
import numpy as np
import os
import pyviennacl as p
import random

import logging
logger = logging.getLogger('pyviennacl')

def noop(*args, **kwargs):
    return os.EX_OK

def diff(a, b):
    p.util.backend_finish()
    ret = 0

    # Convert NumPy types to ViennaCL types (they're much more convenient!)
    if isinstance(a, np.ndarray):
        if a.ndim == 1:
            a = p.Vector(a, dtype = a.dtype)
        elif a.ndim == 2:
            if isinstance(b, p.MagicMethods):
                a = p.Matrix(a, dtype = a.dtype, layout = b.layout)
            else:
                a = p.Matrix(a, dtype = a.dtype)
        else:
            raise TypeError("Something went wrong")

    if isinstance(b, np.ndarray):
        if b.ndim == 1:
            b = p.Vector(b)
        elif b.ndim == 2:
            if isinstance(a, p.MagicMethods):
                b = p.Matrix(b, dtype = b.dtype, layout = a.layout)
            else:
                b = p.Matrix(b, dtype = b.dtype)
        else:
            raise TypeError("Something went wrong")

    # The MagicMethods class guarantees that we have some useful facilities
    #   (and both Node and Leaf are derived from MagicMethods)
    if isinstance(a, p.MagicMethods) and isinstance(b, p.MagicMethods):
        if a.layout != b.layout:
            # We want to make sure that a and b have the same layout
            # So construct a temporary matrix, and assign b to it
            temp = p.Matrix(b.shape, dtype = b.dtype, layout = a.layout)
            p.Assign(temp, b.result).execute()
            b = temp
            log.warning("HERE")
        d = p.ElementFabs(a - b)
        cpu_d = d.as_ndarray()
        if len(d.shape) == 1:
            # vector
            for i in range(d.shape[0]):
                act = math.fabs(cpu_d[i])
                if act > ret:
                    ret = act
        elif len(d.shape) == 2:
            # matrix
            for i in range(d.shape[0]):
                for j in range(d.shape[1]): 
                    act = math.fabs(cpu_d[i, j])
                    if act > ret:
                        ret = act
        else:
            raise TypeError("Something went wrong..")
        return ret
    else:
        # We don't have either ndarrays or ViennaCL types so assume plain scalar
        return math.fabs(a - b) / max(math.fabs(a), math.fabs(b))


def test_vector_slice(test_func,
                      epsilon, dtype,
                      size = 131):
    # Create reference numpy types
    A = np.empty((size,), dtype = dtype)
    big_A = np.ones((size*4,), dtype = dtype) * 3.1415
    B = np.empty((size,), dtype = dtype)
    big_B = np.ones((size*4,), dtype = dtype) * 42.0
    C = np.empty((size,), dtype = dtype)
    big_C = np.ones((size*4,), dtype = dtype) * 2.718

    # Fill A and B with random values
    for i in range(A.shape[0]):
        A[i] = random.random()
    for i in range(B.shape[0]):
        B[i] = random.random()

    # Construct appropriate ViennaCL objects
    vcl_A = p.Vector(A)

    vcl_big_range_A = p.Vector(big_A)
    vcl_big_range_A[size:2*size] = vcl_A
    vcl_range_A = vcl_big_range_A[size:2*size]

    vcl_big_slice_A = p.Vector(big_A)
    vcl_big_slice_A[size:-size:2] = vcl_A
    vcl_slice_A = vcl_big_slice_A[size:-size:2]

    vcl_B = p.Vector(B)

    vcl_big_range_B = p.Vector(big_B)
    vcl_big_range_B[size:2*size] = vcl_B
    vcl_range_B = vcl_big_range_B[size:2*size]

    vcl_big_slice_B = p.Vector(big_B)
    vcl_big_slice_B[size:-size:2] = vcl_B
    vcl_slice_B = vcl_big_slice_B[size:-size:2]

    vcl_C = p.Vector(C)

    vcl_big_range_C = p.Vector(big_C)
    vcl_range_C = vcl_big_range_C[(size - 1):(2*size - 1)]

    vcl_big_slice_C = p.Vector(big_C)
    vcl_slice_C = vcl_big_slice_C[(size - 1):(4*size - 1):3]

    # A=vector, B=vector, C=vector
    print("Now using A=vector, B=vector, C=vector")
    try:
        ret = test_func(epsilon,
                        A, B, C,
                        vcl_A, vcl_B, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Vectors do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=vector, B=vector, C=range
    print("Now using A=vector, B=vector, C=range")
    try:
        ret = test_func(epsilon,
                        A, B, C,
                        vcl_A, vcl_B, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Vectors do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=vector, B=vector, C=slice
    #print("Now using A=vector, B=vector, C=slice")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_A, vcl_B, vcl_slice_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=vector, B=range, C=vector
    print("Now using A=vector, B=range, C=vector")
    try:
        ret = test_func(epsilon,
                        A, B, C,
                        vcl_A, vcl_range_B, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Vectors do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=vector, B=range, C=range
    print("Now using A=vector, B=range, C=range")
    try:
        ret = test_func(epsilon,
                        A, B, C,
                        vcl_A, vcl_range_B, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Vectors do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=vector, B=range, C=slice
    #print("Now using A=vector, B=range, C=slice")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_A, vcl_range_B, vcl_slice_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=vector, B=slice, C=vector
    #print("Now using A=vector, B=slice, C=vector")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_A, vcl_slice_B, vcl_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=vector, B=slice, C=range
    #print("Now using A=vector, B=slice, C=range")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_A, vcl_slice_B, vcl_range_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=vector, B=slice, C=slice
    #print("Now using A=vector, B=slice, C=slice")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_A, vcl_slice_B, vcl_slice_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=vector, C=vector
    print("Now using A=range, B=vector, C=vector")
    try:
        ret = test_func(epsilon,
                        A, B, C,
                        vcl_range_A, vcl_B, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Vectors do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=vector, C=range
    print("Now using A=range, B=vector, C=range")
    try:
        ret = test_func(epsilon,
                        A, B, C,
                        vcl_range_A, vcl_B, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Vectors do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=vector, C=slice
    #print("Now using A=range, B=vector, C=slice")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_range_A, vcl_B, vcl_slice_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=range, C=vector
    print("Now using A=range, B=range, C=vector")
    try:
        ret = test_func(epsilon,
                        A, B, C,
                        vcl_range_A, vcl_range_B, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Vectors do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=range, C=range
    print("Now using A=range, B=range, C=range")
    try:
        ret = test_func(epsilon,
                        A, B, C,
                        vcl_range_A, vcl_range_B, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Vectors do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=range, C=slice
    #print("Now using A=range, B=range, C=slice")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_range_A, vcl_range_B, vcl_slice_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=slice, C=vector
    #print("Now using A=range, B=slice, C=vector")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_range_A, vcl_slice_B, vcl_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=slice, C=range
    #print("Now using A=range, B=slice, C=range")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_range_A, vcl_slice_B, vcl_range_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=slice, C=slice
    #print("Now using A=range, B=slice, C=slice")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_range_A, vcl_slice_B, vcl_slice_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=vector, C=vector
    #print("Now using A=slice, B=vector, C=vector")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_slice_A, vcl_B, vcl_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=vector, C=range
    #print("Now using A=slice, B=vector, C=range")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_slice_A, vcl_B, vcl_range_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=vector, C=slice
    #print("Now using A=slice, B=vector, C=slice")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_slice_A, vcl_B, vcl_slice_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=range, C=vector
    #print("Now using A=slice, B=range, C=vector")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_slice_A, vcl_range_B, vcl_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=range, C=range
    #print("Now using A=slice, B=range, C=range")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_slice_A, vcl_range_B, vcl_range_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=range, C=slice
    #print("Now using A=slice, B=range, C=slice")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_slice_A, vcl_range_B, vcl_slice_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=slice, C=vector
    #print("Now using A=slice, B=slice, C=vector")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_slice_A, vcl_slice_B, vcl_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=slice, C=range
    #print("Now using A=slice, B=slice, C=range")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_slice_A, vcl_slice_B, vcl_range_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=slice, C=slice
    #print("Now using A=slice, B=slice, C=slice")
    #try:
    #    ret = test_func(epsilon,
    #                    A, B, C,
    #                    vcl_slice_A, vcl_slice_B, vcl_slice_C,
    #                    dtype = dtype)
    #except TypeError as e:
    #    if not "Vectors do not have the same layout" in e.args[0]:
    #        raise
    #    else:
    #        logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    return os.EX_OK


def test_matrix_slice(test_func,
                      epsilon, dtype,
                      A_layout = p.ROW_MAJOR, B_layout = p.ROW_MAJOR, 
                      C_layout = p.ROW_MAJOR,
                      size1 = 131, size2 = 67, size3 = 73):
    if A_layout == p.ROW_MAJOR:
        A_order = 'C'
    else:
        A_order = 'F'

    if B_layout == p.ROW_MAJOR:
        B_order = 'C'
    else:
        B_order = 'F'

    if C_layout == p.ROW_MAJOR:
        C_order = 'C'
    else:
        C_order = 'F'

    # Create reference numpy types
    A = np.empty((size1, size2), dtype = dtype, order = A_order)
    big_A = np.ones((size1 * 4, size2 * 4), dtype = dtype, order = A_order) * 3.1415
    B = np.empty((size2, size3), dtype = dtype, order = B_order)
    big_B = np.ones((size2 * 4, size3 * 4), dtype = dtype, order = B_order) * 42.0
    C = np.empty((size1, size3), dtype = dtype, order = C_order)
    big_C = np.ones((size1 * 4, size3 * 4), dtype = dtype, order = C_order) * 2.718

    # Fill A and B with random values
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = random.random()
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] = random.random()

    A_trans = A.T
    big_A_trans = big_A.T
    B_trans = B.T
    big_B_trans = big_B.T
    C_trans = C.T
    big_C_trans = big_C.T

    # Construct appropriate ViennaCL objects
    vcl_A = p.Matrix(A, layout = A_layout)

    vcl_big_range_A = p.Matrix(big_A, layout = A_layout)
    vcl_big_range_A[size1:2*size1, size2:2*size2] = vcl_A
    vcl_range_A = vcl_big_range_A[size1:2*size1, size2:2*size2]

    vcl_big_slice_A = p.Matrix(big_A, layout = A_layout)
    vcl_big_slice_A[size1:-size1:2, size2::3] = vcl_A
    vcl_slice_A = vcl_big_slice_A[size1:-size1:2, size2::3]

    vcl_A_trans = p.Matrix(A_trans, layout = A_layout)

    vcl_big_range_A_trans = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_range_A_trans[size2:2*size2, size1:2*size1] = vcl_A_trans
    vcl_range_A_trans = vcl_big_range_A_trans[size2:2*size2, size1:2*size1]

    vcl_big_slice_A_trans = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_slice_A_trans[size2:-size2:2, size1::3] = vcl_A_trans
    vcl_slice_A_trans = vcl_big_slice_A_trans[size2:-size2:2, size1::3]

    vcl_B = p.Matrix(B, layout = B_layout)

    vcl_big_range_B = p.Matrix(big_B, layout = B_layout)
    vcl_big_range_B[size2:2*size2, size3:2*size3] = vcl_B
    vcl_range_B = vcl_big_range_B[size2:2*size2, size3:2*size3]

    vcl_big_slice_B = p.Matrix(big_B, layout = B_layout)
    vcl_big_slice_B[size2:-size2:2, size3::3] = vcl_B
    vcl_slice_B = vcl_big_slice_B[size2:-size2:2, size3::3]

    vcl_B_trans = p.Matrix(B_trans, layout = B_layout)

    vcl_big_range_B_trans = p.Matrix(big_B_trans, layout = B_layout)
    vcl_big_range_B_trans[size3:2*size3, size2:2*size2] = vcl_B_trans
    vcl_range_B_trans = vcl_big_range_B_trans[size3:2*size3, size2:2*size2]

    vcl_big_slice_B_trans = p.Matrix(big_B_trans, layout = B_layout)
    vcl_big_slice_B_trans[size3:-size3:2, size2::3] = vcl_B_trans
    vcl_slice_B_trans = vcl_big_slice_B_trans[size3:-size3:2, size2::3]

    vcl_C = p.Matrix(C, layout = C_layout)

    vcl_big_range_C = p.Matrix(big_C, layout = C_layout)
    vcl_range_C = vcl_big_range_C[(size1 - 1):(2*size1 - 1), (size3 - 1):(2*size3 - 1)]

    vcl_big_slice_C = p.Matrix(big_C, layout = C_layout)
    vcl_slice_C = vcl_big_slice_C[(size1 - 1):(4*size1 - 1):3, (size3 - 1):(4*size3 - 1):3]

    # A=matrix, B=matrix, C=matrix
    print("Now using A=matrix, B=matrix, C=matrix")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_A, vcl_A_trans,
                        vcl_B, vcl_B_trans, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=matrix, B=matrix, C=range
    print("Now using A=matrix, B=matrix, C=range")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_A, vcl_A_trans,
                        vcl_B, vcl_B_trans, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=matrix, B=matrix, C=slice
    print("Now using A=matrix, B=matrix, C=slice")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_A, vcl_A_trans,
                        vcl_B, vcl_B_trans, vcl_slice_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=matrix, B=range, C=matrix
    print("Now using A=matrix, B=range, C=matrix")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_A, vcl_A_trans,
                        vcl_range_B, vcl_range_B_trans, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=matrix, B=range, C=range
    print("Now using A=matrix, B=range, C=range")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_A, vcl_A_trans,
                        vcl_range_B, vcl_range_B_trans, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=matrix, B=range, C=slice
    print("Now using A=matrix, B=range, C=slice")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_A, vcl_A_trans,
                        vcl_range_B, vcl_range_B_trans, vcl_slice_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=matrix, B=slice, C=matrix
    print("Now using A=matrix, B=slice, C=matrix")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_A, vcl_A_trans,
                        vcl_slice_B, vcl_slice_B_trans, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=matrix, B=slice, C=range
    print("Now using A=matrix, B=slice, C=range")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_A, vcl_A_trans,
                        vcl_slice_B, vcl_slice_B_trans, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=matrix, B=slice, C=slice
    print("Now using A=matrix, B=slice, C=slice")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_A, vcl_A_trans,
                        vcl_slice_B, vcl_slice_B_trans, vcl_slice_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=matrix, C=matrix
    print("Now using A=range, B=matrix, C=matrix")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_range_A, vcl_range_A_trans,
                        vcl_B, vcl_B_trans, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=matrix, C=range
    print("Now using A=range, B=matrix, C=range")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_range_A, vcl_range_A_trans,
                        vcl_B, vcl_B_trans, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=matrix, C=slice
    print("Now using A=range, B=matrix, C=slice")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_range_A, vcl_range_A_trans,
                        vcl_B, vcl_B_trans, vcl_slice_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=range, C=matrix
    print("Now using A=range, B=range, C=matrix")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_range_A, vcl_range_A_trans,
                        vcl_range_B, vcl_range_B_trans, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=range, C=range
    print("Now using A=range, B=range, C=range")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_range_A, vcl_range_A_trans,
                        vcl_range_B, vcl_range_B_trans, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=range, C=slice
    print("Now using A=range, B=range, C=slice")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_range_A, vcl_range_A_trans,
                        vcl_range_B, vcl_range_B_trans, vcl_slice_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=slice, C=matrix
    print("Now using A=range, B=slice, C=matrix")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_range_A, vcl_range_A_trans,
                        vcl_slice_B, vcl_slice_B_trans, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=slice, C=range
    print("Now using A=range, B=slice, C=range")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_range_A, vcl_range_A_trans,
                        vcl_slice_B, vcl_slice_B_trans, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=slice, C=slice
    print("Now using A=range, B=slice, C=slice")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_range_A, vcl_range_A_trans,
                        vcl_slice_B, vcl_slice_B_trans, vcl_slice_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=matrix, C=matrix
    print("Now using A=slice, B=matrix, C=matrix")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_slice_A, vcl_slice_A_trans,
                        vcl_B, vcl_B_trans, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=matrix, C=range
    print("Now using A=slice, B=matrix, C=range")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_slice_A, vcl_slice_A_trans,
                        vcl_B, vcl_B_trans, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=matrix, C=slice
    print("Now using A=slice, B=matrix, C=slice")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_slice_A, vcl_slice_A_trans,
                        vcl_B, vcl_B_trans, vcl_slice_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=range, C=matrix
    print("Now using A=slice, B=range, C=matrix")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_slice_A, vcl_slice_A_trans,
                        vcl_range_B, vcl_range_B_trans, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=range, C=range
    print("Now using A=slice, B=range, C=range")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_slice_A, vcl_slice_A_trans,
                        vcl_range_B, vcl_range_B_trans, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=range, C=slice
    print("Now using A=slice, B=range, C=slice")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_slice_A, vcl_slice_A_trans,
                        vcl_range_B, vcl_range_B_trans, vcl_slice_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=slice, C=matrix
    print("Now using A=slice, B=slice, C=matrix")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_slice_A, vcl_slice_A_trans,
                        vcl_slice_B, vcl_slice_B_trans, vcl_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=slice, C=range
    print("Now using A=slice, B=slice, C=range")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_slice_A, vcl_slice_A_trans,
                        vcl_slice_B, vcl_slice_B_trans, vcl_range_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=slice, C=slice
    print("Now using A=slice, B=slice, C=slice")
    try:
        ret = test_func(epsilon,
                        A, A_trans, B, B_trans, C,
                        vcl_slice_A, vcl_slice_A_trans,
                        vcl_slice_B, vcl_slice_B_trans, vcl_slice_C,
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    return os.EX_OK


def test_matrix_solvers(test_func,
                        epsilon, dtype,
                        A_layout = p.ROW_MAJOR, B_layout = p.ROW_MAJOR, 
                        C_layout = p.ROW_MAJOR,
                        size1 = 131, size2 = 131, size3 = 99):

    if size1 != size2:
        raise AttributeError("size1 must equal size2 for solving a system: %d and %d" % (size1, size2))

    if A_layout == p.ROW_MAJOR:
        A_order = 'C'
    else:
        A_order = 'F'

    if B_layout == p.ROW_MAJOR:
        B_order = 'C'
    else:
        B_order = 'F'

    if C_layout == p.ROW_MAJOR:
        C_order = 'C'
    else:
        C_order = 'F'

    # Create reference numpy types
    A = np.empty((size1, size2), dtype = dtype, order = A_order)
    big_A = np.ones((size1 * 4, size2 * 4), dtype = dtype, order = A_order) * 3.1415

    B = np.empty((size2, size3), dtype = dtype, order = B_order)
    big_B = np.ones((size2 * 4, size3 * 4), dtype = dtype, order = B_order) * 42.0

    # Fill A and B with random values
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A[i, j] = random.random()
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] = random.random()

    A_trans = A.T
    big_A_trans = big_A.T
    B_trans = B.T
    big_B_trans = big_B.T

    A_upper = A.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if j < i:
                A_upper[i, j] = 0

    A_lower = A.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if j > i:
                A_lower[i, j] = 0

    A_unit_upper = A_upper.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i == j:
                A_unit_upper[i, j] = 1.0

    A_unit_lower = A_lower.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i == j:
                A_unit_lower[i, j] = 1.0

    A_trans_upper = A_trans.copy()
    for i in range(A_trans.shape[0]):
        for j in range(A_trans.shape[1]):
            if j < i:
                A_trans_upper[i, j] = 0

    A_trans_lower = A_trans.copy()
    for i in range(A_trans.shape[0]):
        for j in range(A_trans.shape[1]):
            if j > i:
                A_trans_lower[i, j] = 0

    A_trans_unit_upper = A_trans_upper.copy()
    for i in range(A_trans.shape[0]):
        for j in range(A_trans.shape[1]):
            if i == j:
                A_trans_unit_upper[i, j] = 1.0

    A_trans_unit_lower = A_trans_lower.copy()
    for i in range(A_trans.shape[0]):
        for j in range(A_trans.shape[1]):
            if i == j:
                A_trans_unit_lower[i, j] = 1.0

    # Construct appropriate ViennaCL objects

    #  -- A_upper
    vcl_A_upper = p.Matrix(A_upper, layout = A_layout)

    vcl_big_range_A_upper = p.Matrix(big_A, layout = A_layout)
    vcl_big_range_A_upper[size1:2*size1, size2:2*size2] = vcl_A_upper
    vcl_range_A_upper = vcl_big_range_A_upper[size1:2*size1, size2:2*size2]

    vcl_big_slice_A_upper = p.Matrix(big_A, layout = A_layout)
    vcl_big_slice_A_upper[size1:-size1:2, size2::3] = vcl_A_upper
    vcl_slice_A_upper = vcl_big_slice_A_upper[size1:-size1:2, size2::3]

    vcl_A_trans_upper = p.Matrix(A_trans_upper, layout = A_layout)

    vcl_big_range_A_trans_upper = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_range_A_trans_upper[size2:2*size2, size1:2*size1] = vcl_A_trans_upper
    vcl_range_A_trans_upper = vcl_big_range_A_trans_upper[size2:2*size2, size1:2*size1]

    vcl_big_slice_A_trans_upper = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_slice_A_trans_upper[size2:-size2:2, size1::3] = vcl_A_trans_upper
    vcl_slice_A_trans_upper = vcl_big_slice_A_trans_upper[size2:-size2:2, size1::3]

    #  -- A_unit_upper
    vcl_A_unit_upper = p.Matrix(A_unit_upper, layout = A_layout)

    vcl_big_range_A_unit_upper = p.Matrix(big_A, layout = A_layout)
    vcl_big_range_A_unit_upper[size1:2*size1, size2:2*size2] = vcl_A_unit_upper
    vcl_range_A_unit_upper = vcl_big_range_A_unit_upper[size1:2*size1, size2:2*size2]

    vcl_big_slice_A_unit_upper = p.Matrix(big_A, layout = A_layout)
    vcl_big_slice_A_unit_upper[size1:-size1:2, size2::3] = vcl_A_unit_upper
    vcl_slice_A_unit_upper = vcl_big_slice_A_unit_upper[size1:-size1:2, size2::3]

    vcl_A_trans_unit_upper = p.Matrix(A_trans_unit_upper, layout = A_layout)

    vcl_big_range_A_trans_unit_upper = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_range_A_trans_unit_upper[size2:2*size2, size1:2*size1] = vcl_A_trans_unit_upper
    vcl_range_A_trans_unit_upper = vcl_big_range_A_trans_unit_upper[size2:2*size2, size1:2*size1]

    vcl_big_slice_A_trans_unit_upper = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_slice_A_trans_unit_upper[size2:-size2:2, size1::3] = vcl_A_trans_unit_upper
    vcl_slice_A_trans_unit_upper = vcl_big_slice_A_trans_unit_upper[size2:-size2:2, size1::3]

    #  -- A_lower
    vcl_A_lower = p.Matrix(A_lower, layout = A_layout)

    vcl_big_range_A_lower = p.Matrix(big_A, layout = A_layout)
    vcl_big_range_A_lower[size1:2*size1, size2:2*size2] = vcl_A_lower
    vcl_range_A_lower = vcl_big_range_A_lower[size1:2*size1, size2:2*size2]

    vcl_big_slice_A_lower = p.Matrix(big_A, layout = A_layout)
    vcl_big_slice_A_lower[size1:-size1:2, size2::3] = vcl_A_lower
    vcl_slice_A_lower = vcl_big_slice_A_lower[size1:-size1:2, size2::3]

    vcl_A_trans_lower = p.Matrix(A_trans_lower, layout = A_layout)

    vcl_big_range_A_trans_lower = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_range_A_trans_lower[size2:2*size2, size1:2*size1] = vcl_A_trans_lower
    vcl_range_A_trans_lower = vcl_big_range_A_trans_lower[size2:2*size2, size1:2*size1]

    vcl_big_slice_A_trans_lower = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_slice_A_trans_lower[size2:-size2:2, size1::3] = vcl_A_trans_lower
    vcl_slice_A_trans_lower = vcl_big_slice_A_trans_lower[size2:-size2:2, size1::3]

    #  -- A_unit_lower
    vcl_A_unit_lower = p.Matrix(A_unit_lower, layout = A_layout)

    vcl_big_range_A_unit_lower = p.Matrix(big_A, layout = A_layout)
    vcl_big_range_A_unit_lower[size1:2*size1, size2:2*size2] = vcl_A_unit_lower
    vcl_range_A_unit_lower = vcl_big_range_A_unit_lower[size1:2*size1, size2:2*size2]

    vcl_big_slice_A_unit_lower = p.Matrix(big_A, layout = A_layout)
    vcl_big_slice_A_unit_lower[size1:-size1:2, size2::3] = vcl_A_unit_lower
    vcl_slice_A_unit_lower = vcl_big_slice_A_unit_lower[size1:-size1:2, size2::3]

    vcl_A_trans_unit_lower = p.Matrix(A_trans_unit_lower, layout = A_layout)

    vcl_big_range_A_trans_unit_lower = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_range_A_trans_unit_lower[size2:2*size2, size1:2*size1] = vcl_A_trans_unit_lower
    vcl_range_A_trans_unit_lower = vcl_big_range_A_trans_unit_lower[size2:2*size2, size1:2*size1]

    vcl_big_slice_A_trans_unit_lower = p.Matrix(big_A_trans, layout = A_layout)
    vcl_big_slice_A_trans_unit_lower[size2:-size2:2, size1::3] = vcl_A_trans_unit_lower
    vcl_slice_A_trans_unit_lower = vcl_big_slice_A_trans_unit_lower[size2:-size2:2, size1::3]

    #  -- B
    vcl_B = p.Matrix(B, layout = B_layout)

    vcl_big_range_B = p.Matrix(big_B, layout = B_layout)
    vcl_big_range_B[size2:2*size2, size3:2*size3] = vcl_B
    vcl_range_B = vcl_big_range_B[size2:2*size2, size3:2*size3]

    vcl_big_slice_B = p.Matrix(big_B, layout = B_layout)
    vcl_big_slice_B[size2:-size2:2, size3::3] = vcl_B
    vcl_slice_B = vcl_big_slice_B[size2:-size2:2, size3::3]

    vcl_B_trans = p.Matrix(B_trans, layout = B_layout)

    vcl_big_range_B_trans = p.Matrix(big_B_trans, layout = B_layout)
    vcl_big_range_B_trans[size3:2*size3, size2:2*size2] = vcl_B_trans
    vcl_range_B_trans = vcl_big_range_B_trans[size3:2*size3, size2:2*size2]

    vcl_big_slice_B_trans = p.Matrix(big_B_trans, layout = B_layout)
    vcl_big_slice_B_trans[size3:-size3:2, size2::3] = vcl_B_trans
    vcl_slice_B_trans = vcl_big_slice_B_trans[size3:-size3:2, size2::3]

    # A=matrix, B=matrix
    print("Now using A=matrix, B=matrix")
    try:
        ret = test_func(epsilon,
                        (A_upper, A_unit_upper, A_lower, A_unit_lower,
                         A_trans_upper, A_trans_unit_upper,
                         A_trans_lower, A_trans_unit_lower),
                        (B, B_trans),
                        (vcl_A_upper, vcl_A_unit_upper,
                         vcl_A_lower, vcl_A_unit_lower,
                         vcl_A_trans_upper, vcl_A_trans_unit_upper,
                         vcl_A_trans_lower, vcl_A_trans_unit_lower),
                        (vcl_B, vcl_B_trans),
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=matrix, B=range
    print("Now using A=matrix, B=range")
    try:
        ret = test_func(epsilon,
                        (A_upper, A_unit_upper, A_lower, A_unit_lower,
                         A_trans_upper, A_trans_unit_upper,
                         A_trans_lower, A_trans_unit_lower),
                        (B, B_trans),
                        (vcl_A_upper, vcl_A_unit_upper,
                         vcl_A_lower, vcl_A_unit_lower,
                         vcl_A_trans_upper, vcl_A_trans_unit_upper,
                         vcl_A_trans_lower, vcl_A_trans_unit_lower),
                        (vcl_range_B, vcl_range_B_trans),
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=matrix, B=slice
    print("Now using A=matrix, B=slice")
    try:
        ret = test_func(epsilon,
                        (A_upper, A_unit_upper, A_lower, A_unit_lower,
                         A_trans_upper, A_trans_unit_upper,
                         A_trans_lower, A_trans_unit_lower),
                        (B, B_trans),
                        (vcl_A_upper, vcl_A_unit_upper,
                         vcl_A_lower, vcl_A_unit_lower,
                         vcl_A_trans_upper, vcl_A_trans_unit_upper,
                         vcl_A_trans_lower, vcl_A_trans_unit_lower),
                        (vcl_slice_B, vcl_slice_B_trans),
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=matrix
    print("Now using A=range, B=matrix")
    try:
        ret = test_func(epsilon,
                        (A_upper, A_unit_upper, A_lower, A_unit_lower,
                         A_trans_upper, A_trans_unit_upper,
                         A_trans_lower, A_trans_unit_lower),
                        (B, B_trans),
                        (vcl_range_A_upper, vcl_range_A_unit_upper,
                         vcl_range_A_lower, vcl_range_A_unit_lower,
                         vcl_range_A_trans_upper,vcl_range_A_trans_unit_upper,
                         vcl_range_A_trans_lower,vcl_range_A_trans_unit_lower),
                        (vcl_B, vcl_B_trans),
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=range
    print("Now using A=range, B=range")
    try:
        ret = test_func(epsilon,
                        (A_upper, A_unit_upper, A_lower, A_unit_lower,
                         A_trans_upper, A_trans_unit_upper,
                         A_trans_lower, A_trans_unit_lower),
                        (B, B_trans),
                        (vcl_range_A_upper, vcl_range_A_unit_upper,
                         vcl_range_A_lower, vcl_range_A_unit_lower,
                         vcl_range_A_trans_upper,vcl_range_A_trans_unit_upper,
                         vcl_range_A_trans_lower,vcl_range_A_trans_unit_lower),
                        (vcl_range_B, vcl_range_B_trans),
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=range, B=slice
    print("Now using A=range, B=slice")
    try:
        ret = test_func(epsilon,
                        (A_upper, A_unit_upper, A_lower, A_unit_lower,
                         A_trans_upper, A_trans_unit_upper,
                         A_trans_lower, A_trans_unit_lower),
                        (B, B_trans),
                        (vcl_range_A_upper, vcl_range_A_unit_upper,
                         vcl_range_A_lower, vcl_range_A_unit_lower,
                         vcl_range_A_trans_upper,vcl_range_A_trans_unit_upper,
                         vcl_range_A_trans_lower,vcl_range_A_trans_unit_lower),
                        (vcl_slice_B, vcl_slice_B_trans),
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=matrix
    print("Now using A=slice, B=matrix")
    try:
        ret = test_func(epsilon,
                        (A_upper, A_unit_upper, A_lower, A_unit_lower,
                         A_trans_upper, A_trans_unit_upper,
                         A_trans_lower, A_trans_unit_lower),
                        (B, B_trans),
                        (vcl_slice_A_upper, vcl_slice_A_unit_upper,
                         vcl_slice_A_lower, vcl_slice_A_unit_lower,
                         vcl_slice_A_trans_upper,vcl_slice_A_trans_unit_upper,
                         vcl_slice_A_trans_lower,vcl_slice_A_trans_unit_lower),
                        (vcl_B, vcl_B_trans),
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=range
    print("Now using A=slice, B=range")
    try:
        ret = test_func(epsilon,
                        (A_upper, A_unit_upper, A_lower, A_unit_lower,
                         A_trans_upper, A_trans_unit_upper,
                         A_trans_lower, A_trans_unit_lower),
                        (B, B_trans),
                        (vcl_slice_A_upper, vcl_slice_A_unit_upper,
                         vcl_slice_A_lower, vcl_slice_A_unit_lower,
                         vcl_slice_A_trans_upper,vcl_slice_A_trans_unit_upper,
                         vcl_slice_A_trans_lower,vcl_slice_A_trans_unit_lower),
                        (vcl_range_B, vcl_range_B_trans),
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    # A=slice, B=slice
    print("Now using A=slice, B=slice")
    try:
        ret = test_func(epsilon,
                        (A_upper, A_unit_upper, A_lower, A_unit_lower,
                         A_trans_upper, A_trans_unit_upper,
                         A_trans_lower, A_trans_unit_lower),
                        (B, B_trans),
                        (vcl_slice_A_upper, vcl_slice_A_unit_upper,
                         vcl_slice_A_lower, vcl_slice_A_unit_lower,
                         vcl_slice_A_trans_upper,vcl_slice_A_trans_unit_upper,
                         vcl_slice_A_trans_lower,vcl_slice_A_trans_unit_lower),
                        (vcl_slice_B, vcl_slice_B_trans),
                        dtype = dtype)
    except TypeError as e:
        if not "Matrices do not have the same layout" in e.args[0]:
            raise
        else:
            logger.debug("EXCEPTION EXECUTING was: %s" % e.args[0])

    return os.EX_OK


def test_matrix_layout(test_func, test_kernel, epsilon, dtype,
                       size1 = 131, size2 = 131, size3 = 73,
                       num_matrices = 3):
    # A=row, B=row, C=row
    print("///////////////////////////////////////")
    print("/// Now testing A=row, B=row, C=row ///")
    print("///////////////////////////////////////")
    test_func(test_kernel, epsilon, dtype, p.ROW_MAJOR, p.ROW_MAJOR, p.ROW_MAJOR, size1, size2, size3)

    if num_matrices == 3:
        # A=row, B=row, C=col
        print("///////////////////////////////////////")
        print("/// Now testing A=row, B=row, C=col ///")
        print("///////////////////////////////////////")
        test_func(test_kernel, epsilon, dtype, p.ROW_MAJOR, p.ROW_MAJOR, p.COL_MAJOR, size1, size2, size3)

    # A=row, B=col, C=row
    print("///////////////////////////////////////")
    print("/// Now testing A=row, B=col, C=row ///")
    print("///////////////////////////////////////")
    test_func(test_kernel, epsilon, dtype, p.ROW_MAJOR, p.COL_MAJOR, p.ROW_MAJOR, size1, size2, size3)

    if num_matrices == 3:
        # A=row, B=col, C=col
        print("///////////////////////////////////////")
        print("/// Now testing A=row, B=col, C=col ///")
        print("///////////////////////////////////////")
        test_func(test_kernel, epsilon, dtype, p.ROW_MAJOR, p.COL_MAJOR, p.COL_MAJOR, size1, size2, size3)

    # A=col, B=row, C=row
    print("///////////////////////////////////////")
    print("/// Now testing A=col, B=row, C=row ///")
    print("///////////////////////////////////////")
    test_func(test_kernel, epsilon, dtype, p.COL_MAJOR, p.ROW_MAJOR, p.ROW_MAJOR, size1, size2, size3)

    if num_matrices == 3:
        # A=col, B=row, C=col
        print("///////////////////////////////////////")
        print("/// Now testing A=col, B=row, C=col ///")
        print("///////////////////////////////////////")
        test_func(test_kernel, epsilon, dtype, p.COL_MAJOR, p.ROW_MAJOR, p.COL_MAJOR, size1, size2, size3)

    # A=col, B=col, C=row
    print("///////////////////////////////////////")
    print("/// Now testing A=col, B=col, C=row ///")
    print("///////////////////////////////////////")
    test_func(test_kernel, epsilon, dtype, p.COL_MAJOR, p.COL_MAJOR, p.ROW_MAJOR, size1, size2, size3)
    
    if num_matrices == 3:
        # A=col, B=col, C=col
        print("///////////////////////////////////////")
        print("/// Now testing A=col, B=col, C=col ///")
        print("///////////////////////////////////////")
        test_func(test_kernel, epsilon, dtype, p.COL_MAJOR, p.COL_MAJOR, p.COL_MAJOR, size1, size2, size3)

    return os.EX_OK
