from __future__ import division

import pyviennacl as p
import numpy as np
from _common import *
from itertools import product

layouts = [p.ROW_MAJOR, p.COL_MAJOR]
matrix_getters = [('matrix', 'get_matrix'),
                  ('matrix_range', 'get_matrix_range'),
                  ('matrix_slice', 'get_matrix_slice'),
                  ('matrix_trans', 'get_matrix_trans'),
                  ('matrix_range_trans', 'get_matrix_range_trans'),
                  ('matrix_slice_trans', 'get_matrix_slice_trans')]
vector_getters = [('vector', 'get_vector'),
                  ('vector_range', 'get_vector_range'),
                  ('vector_slice', 'get_vector_slice')]
scalar_getters = [('host_scalar', 'get_host_scalar'),
                  ('device_scalar', 'get_device_scalar')]

if double_support:
    dtype_tolerances = [('float32', 1.0E-3), ('float64', 1.0E-11)]
else:
    dtype_tolerances = [('float32', 1.0E-3)]

A_matrix_operations = [
    #('abs', 'np.absolute', 'p.absolute'), # Only for int types!
    ('acos', 'np.arccos', 'p.arccos'),
    ('asin', 'np.arcsin', 'p.arcsin'),
    ('atan', 'np.arctan', 'p.arctan'),
    ('ceil', 'np.ceil', 'p.ceil'),
    ('cos', 'np.cos', 'p.cos'),
    ('cosh', 'np.cosh', 'p.cosh'),
    ('exp', 'np.exp', 'p.exp'),
    ('fabs', 'np.fabs', 'p.fabs'),
    ('floor', 'np.floor', 'p.floor'),
    ('log', 'np.log', 'p.log'),
    ('log10', 'np.log10', 'p.log10'),
    ('sin', 'np.sin', 'p.sin'),
    ('sinh', 'np.sinh', 'p.sinh'),
    ('sqrt', 'np.sqrt', 'p.sqrt'),
    ('tan', 'np.tan', 'p.tan'),
    ('tanh', 'np.tanh', 'p.tanh')
]

Ap_matrix_operations = [
    ('mul', 'mul', 'mul'),
    ('div', 'div', 'div')
]

Ax_matrix_operations = [
    ('dgemv', 'dot', 'dot')
]

AB_matrix_operations = [
    ('add', 'add', 'add'),
    ('iadd', 'iadd', 'iadd'),
    ('sub', 'sub', 'sub'),
    ('isub', 'isub', 'isub'),

    ('elementwise_pow', 'pow', 'pow'),
    ('elementwise_prod', 'mul', 'p.ElementProd'),
    ('elementwise_div', 'div', 'p.ElementDiv')
 ]

AB_matrix_products = [
    ('dgemm', 'dot', 'dot'),
    ('iadd_dgemm', 'iadd_dot', 'iadd_dot'),
    ('isub_dgemm', 'isub_dot', 'isub_dot'),
 ]

ABp_matrix_operations = [
    ('scaled_add_left1', 'scaled_add_left1', 'scaled_add_left1'),
    ('scaled_add_right1', 'scaled_add_right1', 'scaled_add_right1'),
    ('scaled_add_both1', 'scaled_add_both1', 'scaled_add_both1'),

    ('scaled_sub_left1', 'scaled_sub_left1', 'scaled_sub_left1'),
    ('scaled_sub_right1', 'scaled_sub_right1', 'scaled_sub_right1'),
    ('scaled_sub_both1', 'scaled_sub_both1', 'scaled_sub_both1'),

    ('scaled_add_left2', 'scaled_add_left2', 'scaled_add_left2'),
    ('scaled_add_right2', 'scaled_add_right2', 'scaled_add_right2'),
    ('scaled_add_both2', 'scaled_add_both2', 'scaled_add_both2'),

    ('scaled_sub_left2', 'scaled_sub_left2', 'scaled_sub_left2'),
    ('scaled_sub_right2', 'scaled_sub_right2', 'scaled_sub_right2'),
    ('scaled_sub_both2', 'scaled_sub_both2', 'scaled_sub_both2')
]


A_test_code_header = """
def test_%s_%s_%s_%s():
    size1, size2 = 11, 10
    alpha1 = p.Scalar
"""

Ap_test_code_header = """
def test_%s_%s_%s_%s_%s():
    size1, size2 = 11, 10
    alpha1 = p.Scalar
"""
Ax_test_code_header = Ap_test_code_header
AB_test_code_header = Ap_test_code_header

ABp_test_code_header = """
def test_%s_%s_%s_%s_%s_%s():
    size1, size2 = 11, 10
    alpha1 = p.Scalar
"""

test_code_footer = """
    act_diff = math.fabs(diff(numpy_C, vcl_C))
    assert act_diff <= tol, "diff was {} > tolerance {}".format(act_diff, tol)
"""

for layout_, d_t_, getter1_, op_ in product(layouts, dtype_tolerances, matrix_getters, A_matrix_operations):
    dt = d_t_[0]
    tol = d_t_[1]
    test_code = (A_test_code_header + """
    numpy_A, vcl_A = %s(size1, size2, '%s', dt, None)
    numpy_C = %s(numpy_A)
    vcl_C = %s(vcl_A)
""" + test_code_footer) % (getter1_[0], op_[0], layout_, dt, getter1_[1], layout_, op_[1], op_[2])
    exec(test_code)

for layout_, d_t_, getter1_, getter2_, op_ in product(layouts, dtype_tolerances, matrix_getters, scalar_getters, Ap_matrix_operations):
    dt = d_t_[0]
    tol = d_t_[1]
    test_code = (Ap_test_code_header + """
    numpy_A, vcl_A = %s(size1, size2, '%s', dt, None)
    alpha = %s(dt)
    numpy_C = %s(numpy_A, alpha.value)
    vcl_C = %s(vcl_A, alpha)
""" + test_code_footer) % (getter1_[0], getter2_[0], op_[0], layout_, dt, getter1_[1], layout_, getter2_[1], op_[1], op_[2])
    exec(test_code)

for layout_, d_t_, getter1_, getter2_, op_ in product(layouts, dtype_tolerances, matrix_getters, vector_getters, Ax_matrix_operations):
    dt = d_t_[0]
    tol = d_t_[1]
    test_code = (Ax_test_code_header + """
    numpy_A, vcl_A = %s(size1, size2, '%s', dt, None)
    numpy_x, vcl_x = %s(size2, dt)
    numpy_C = %s(numpy_A, numpy_x)
    vcl_C = %s(vcl_A, vcl_x)
""" + test_code_footer) % (getter1_[0], getter2_[0], op_[0], layout_, dt, getter1_[1], layout_, getter2_[1], op_[1], op_[2])
    exec(test_code)

for layout_, d_t_, getter1_, getter2_, op_ in product(layouts, dtype_tolerances, matrix_getters, matrix_getters, AB_matrix_operations):
    dt = d_t_[0]
    tol = d_t_[1]
    test_code = (AB_test_code_header + """
    numpy_A, vcl_A = %s(size1, size2, '%s', dt, None)
    numpy_B, vcl_B = %s(size1, size2, '%s', dt, None)
    numpy_C = %s(numpy_A, numpy_B)
    vcl_C = %s(vcl_A, vcl_B)
""" + test_code_footer) % (getter1_[0], getter2_[0], op_[0], layout_, dt, getter1_[1], layout_, getter2_[1], layout_, op_[1], op_[2])
    exec(test_code)

for layout_, d_t_, getter1_, getter2_, op_ in product(layouts, dtype_tolerances, matrix_getters, matrix_getters, AB_matrix_products):
    dt = d_t_[0]
    tol = d_t_[1]
    test_code = (AB_test_code_header + """
    numpy_A, vcl_A = %s(size1, size2, '%s', dt, None)
    numpy_B, vcl_B = %s(size2, size1, '%s', dt, None)
    numpy_C = %s(numpy_A, numpy_B)
    vcl_C = %s(vcl_A, vcl_B)
""" + test_code_footer) % (getter1_[0], getter2_[0], op_[0], layout_, dt, getter1_[1], layout_, getter2_[1], layout_, op_[1], op_[2])
    exec(test_code)

for layout_, d_t_, getter1_, getter2_, getter3_, op_ in product(layouts, dtype_tolerances, matrix_getters, matrix_getters, scalar_getters, ABp_matrix_operations):
    dt = d_t_[0]
    tol = d_t_[1]
    test_code = (ABp_test_code_header + """
    numpy_A, vcl_A = %s(size1, size2, '%s', dt, None)
    numpy_B, vcl_B = %s(size1, size2, '%s', dt, None)
    alpha = %s(dt)
    numpy_C = %s(numpy_A, numpy_B, alpha.value)
    vcl_C = %s(vcl_A, vcl_B, alpha)
""" + test_code_footer) % (getter1_[0], getter2_[0], getter3_[0], op_[0], layout_, dt, getter1_[1], layout_, getter2_[1], layout_, getter3_[1], op_[1], op_[2])
    exec(test_code)

