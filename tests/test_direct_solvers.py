from __future__ import division

import pyviennacl as p
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import scipy.sparse.linalg as spspla
from _common import *
from itertools import product

layouts = [p.ROW_MAJOR, p.COL_MAJOR]

if double_support:
    dtype_tolerances = [('float32', 1.0E-3), ('float64', 1.0E-8)]
else:
    dtype_tolerances = [('float32', 1.0E-3)]

matrix_getters = [('matrix', 'get_matrix'),
                  ('matrix_range', 'get_matrix_range'),
                  ('matrix_slice', 'get_matrix_slice'),
                  ('matrix_trans', 'get_matrix_trans'),
                  ('matrix_range_trans', 'get_matrix_range_trans'),
                  ('matrix_slice_trans', 'get_matrix_slice_trans')]
vector_getters = [('vector', 'get_vector'),
                  ('vector_range', 'get_vector_range'),
                  ('vector_slice', 'get_vector_slice')]

forms = ['upper', 'unit_upper', 'lower', 'unit_lower']
forms_tags = {
    'upper': p.tags.Upper(),
    'unit_upper': p.tags.UnitUpper(),
    'lower': p.tags.Lower(),
    'unit_lower': p.tags.UnitLower()
}

for layout_, d_t_, form_, getter1_, getter2_ in product(layouts, dtype_tolerances, forms, matrix_getters, matrix_getters):
    dt_ = d_t_[0]
    tol_ = d_t_[1]

    def A_solve_B_test_factory(A_form, dt, tol, layout, getter1, getter2):
        def _test():
            size1, size2 = 6, 6
            numpy_A, vcl_A = getter1(size1, size2, layout, dt, A_form)
            numpy_B, vcl_B = getter2(size1, size2, layout, dt)
            vcl_X = p.solve(vcl_A, vcl_B, forms_tags[A_form])
            numpy_X = npla.solve(numpy_A, numpy_B)
            act_diff = math.fabs(diff(numpy_X, vcl_X))
            assert act_diff <= tol, "diff was {} > tolerance {}".format(act_diff, tol)
        return _test
        
    exec("test_%s_%s_A_%s_B_%s_%s = A_solve_B_test_factory('%s', p.%s, %g, '%s', %s, %s)" % (getter1_[0], form_, getter2_[0], layout_, dt_, form_, dt_, tol_, layout_, getter1_[1], getter2_[1]))

for layout_, d_t_, form_, getter1_, getter2_ in product(layouts, dtype_tolerances, forms, matrix_getters, vector_getters):
    dt_ = d_t_[0]
    tol_ = d_t_[1]

    def A_solve_b_test_factory(A_form, dt, tol, layout, getter1, getter2):
        def _test():
            size1, size2 = 6, 6
            numpy_A, vcl_A = getter1(size1, size2, layout, dt, A_form)
            numpy_b, vcl_b = getter2(size1, dt)
            vcl_x = p.solve(vcl_A, vcl_b, forms_tags[A_form])
            numpy_x = npla.solve(numpy_A, numpy_b)
            act_diff = math.fabs(diff(numpy_x, vcl_x))
            print(getter1.__name__, getter2.__name__, A_form)
            assert act_diff <= tol, "diff was {} > tolerance {}".format(act_diff, tol)
        return _test
        
    exec("test_%s_%s_A_%s_b_%s_%s = A_solve_b_test_factory('%s', p.%s, %g, '%s', %s, %s)" % (getter1_[0], form_, getter2_[0], layout_, dt_, form_, dt_, tol_, layout_, getter1_[1], getter2_[1]))

