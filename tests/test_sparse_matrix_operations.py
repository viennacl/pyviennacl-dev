from __future__ import division

import math
import numpy as np
import pyviennacl as p
import scipy.sparse.linalg as spspla
from _common import *
from itertools import product

size, sparsity = 20, 0.1

dtype_tolerances = [('float32', 1.0E-3), ('float64', 1.0E-8)]

dense_matrix_getters = [('matrix', 'get_matrix'),
                        ('matrix_range', 'get_matrix_range'),
                        ('matrix_slice', 'get_matrix_slice'),
                        ('matrix_trans', 'get_matrix_trans'),
                        ('matrix_range_trans', 'get_matrix_range_trans'),
                        ('matrix_slice_trans', 'get_matrix_slice_trans')]

sparse_matrix_types = [('compressed_matrix', 'p.CompressedMatrix'),
                       ('coordinate_matrix', 'p.CoordinateMatrix'),
                       ('ell_matrix', 'p.ELLMatrix'),
                       ('hyb_matrix', 'p.HybridMatrix')]

rhs_vector_getters = [('vector', 'get_vector')] #,
                      #('vector_range', 'get_vector_range'),
                      #('vector_slice', 'get_vector_slice')]

Ax_matrix_operations = [
    ('dgemv', 'dot', 'dot')
]

for op_, d_t_, sparse_type_, vector_getter_ in product(Ax_matrix_operations, dtype_tolerances, sparse_matrix_types, rhs_vector_getters):
    dt_ = d_t_[0]
    tol_ = d_t_[1]

    def factory(dt, tol, sparse_type, vector_getter, numpy_op, vcl_op):
        def _test():
            vcl_A = get_sparse_matrix(size, sparsity, dt, sparse_type)
            numpy_A = vcl_A.as_ndarray()
            numpy_x, vcl_x = vector_getter(size, dt)

            numpy_b = numpy_op(numpy_A, numpy_x)
            vcl_b = vcl_op(vcl_A, vcl_x)

            # compare with numpy_solution
            act_diff = math.fabs(diff(numpy_b, vcl_b))
            assert act_diff <= tol, "diff was {} > tolerance {}".format(act_diff, tol)

        return _test
        
    exec("test_%s_%s_A_%s_x_%s = factory(p.%s, %g, %s, %s, %s, %s)" % (op_[0], sparse_type_[0], vector_getter_[0], dt_, dt_, tol_, sparse_type_[1], vector_getter_[1], op_[1], op_[2]))

