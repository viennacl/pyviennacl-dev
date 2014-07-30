from __future__ import division

import pyviennacl as p
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
from _common import *
from itertools import product

layouts = [p.ROW_MAJOR, p.COL_MAJOR]
dtype_tolerances = [(p.float32, 1.0E-1), (p.float64, 1.0E-8)]
matrix_getters = [get_matrix, get_matrix_range, get_matrix_slice, get_matrix_trans, get_matrix_range_trans, get_matrix_slice_trans]
vector_getters = [get_vector, get_vector_range, get_vector_slice]

forms = ['upper', 'unit_upper', 'lower', 'unit_lower']
forms_tags = {
    'upper': p.upper_tag(),
    'unit_upper': p.unit_upper_tag(),
    'lower': p.lower_tag(),
    'unit_lower': p.unit_lower_tag()
}

for layout_, d_t_, form_ in product(layouts, dtype_tolerances, forms):
    dt_ = d_t_[0]
    tol_ = d_t_[1]

    def A_solve_B_test_factory(A_form, dt, tol, layout):
        def _test():
            size1, size2 = 6, 6
            for getter1, getter2 in product(matrix_getters, matrix_getters):
                numpy_A, vcl_A = getter1(size1, size2, layout, dt, A_form)
                numpy_B, vcl_B = getter2(size1, size2, layout, dt)
                vcl_X = p.solve(vcl_A, vcl_B, forms_tags[A_form])
                numpy_X = npla.solve(numpy_A, numpy_B)
                act_diff = math.fabs(diff(numpy_X, vcl_X))
                assert act_diff <= tol, "diff was {} > tolerance {}".format(act_diff, tol)
        return _test
        
    exec("test_A_%s_B_%s_%s = A_solve_B_test_factory('%s', p.%s, %g, '%s')" % (form_, layout_, dt_.__name__, form_, dt_.__name__, tol_, layout_))

    def A_solve_b_test_factory(A_form, dt, tol, layout):
        def _test():
            size1, size2 = 6, 6
            for getter1, getter2 in product(matrix_getters, vector_getters):
                numpy_A, vcl_A = getter1(size1, size2, layout, dt, A_form)
                numpy_b, vcl_b = getter2(size1, dt)
                vcl_x = p.solve(vcl_A, vcl_b, forms_tags[A_form])
                numpy_x = spla.solve(numpy_A, numpy_b)
                act_diff = math.fabs(diff(numpy_x, vcl_x))
                print(getter1.__name__, getter2.__name__, A_form)
                assert act_diff <= tol, "diff was {} > tolerance {}".format(act_diff, tol)
        return _test
        
    exec("test_A_%s_b_%s_%s = A_solve_b_test_factory('%s', p.%s, %g, '%s')" % (form_, layout_, dt_.__name__, form_, dt_.__name__, tol_, layout_))

