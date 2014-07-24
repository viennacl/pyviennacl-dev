from __future__ import division

import pyviennacl as p
import numpy as np
from _common import *
from itertools import product

layouts = [p.ROW_MAJOR, p.COL_MAJOR]
matrix_getters = [get_matrix, get_matrix_range, get_matrix_slice]
vector_getters = [get_vector, get_vector_range, get_vector_slice]
scalar_getters = [get_host_scalar, get_device_scalar]
dtype_tolerances = [(p.float32, 1.0E-3), (p.float64, 1.0E-11)]


A_matrix_operations = [
#    ('abs', 'np.absolute', 'p.absolute'),
#    ('acos', 'np.arccos', 'p.arctan'),
#    ('asin', 'np.arcsin', 'p.arcsin'),
#    ('atan', 'np.arctan', 'p.arccos'),
#    ('ceil', 'np.ceil', 'p.ceil'),
#    ('cos', 'np.cos', 'p.cos'),
#    ('cosh', 'np.cosh', 'p.cosh'),
#    ('exp', 'np.exp', 'p.exp'),
#    ('fabs', 'np.fabs', 'p.fabs'),
#    ('floor', 'np.floor', 'p.floor'),
#    ('log', 'np.log', 'p.log'),
#    ('log10', 'np.log10', 'p.log10'),
#    ('sin', 'np.sin', 'p.sin'),
#    ('sinh', 'np.sinh', 'p.sinh'),
#    ('sqrt', 'np.sqrt', 'p.sqrt'),
#    ('tan', 'np.tan', 'p.tan'),
#    ('tanh', 'np.tanh', 'p.tanh')
]

Ap_matrix_operations = [
    ('mul', 'mul', 'mul'),
    ('div', 'div', 'div')
]

Ax_matrix_operations = [
    ('dot_vector', 'dot', 'dot')
]

AB_matrix_operations = [
    ('add', 'add', 'add'),
    ('iadd', 'iadd', 'iadd'),
    ('sub', 'sub', 'sub'),
    ('isub', 'isub', 'isub'),

    #('elementwise_pow', 'pow', 'pow'),
    ('elementwise_prod', 'mul', 'p.ElementProd'),
    ('elementwise_div', 'div', 'p.ElementDiv')
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


test_code_header = """
def test_%s_%s_%s():
    size1, size2 = 11, 10
    alpha1 = p.Scalar
"""

test_code_footer = """
        act_diff = math.fabs(diff(numpy_C, vcl_C))
        assert act_diff <= tol, "diff was {} > tolerance {}".format(act_diff, tol)
"""

for layout, d_t in product(layouts, dtype_tolerances):
    dt = d_t[0]
    tol = d_t[1]

    for op in A_matrix_operations:
        test_code = (test_code_header + """
    for getter1 in matrix_getters:
        numpy_A, vcl_A = getter1(size1, size2, layout, dt, None)
        numpy_C = %s(numpy_A)
        vcl_C = %s(vcl_A)
""" + test_code_footer) % (op[0], layout, dt.__name__, op[1], op[2])
        exec(test_code)

    for op in Ap_matrix_operations:
        test_code = (test_code_header + """
    for getter1, getter2 in product(matrix_getters, scalar_getters):
        numpy_A, vcl_A = getter1(size1, size2, layout, dt, None)
        alpha = getter2(dt)
        numpy_C = %s(numpy_A, alpha.value)
        vcl_C = %s(vcl_A, alpha)
""" + test_code_footer) % (op[0], layout, dt.__name__, op[1], op[2])
        exec(test_code)

    for op in Ax_matrix_operations:
        test_code = (test_code_header + """
    for getter1, getter2 in product(matrix_getters, vector_getters):
        numpy_A, vcl_A = getter1(size1, size2, layout, dt, None)
        numpy_x, vcl_x = getter2(size2, dt)
        numpy_C = %s(numpy_A, numpy_x)
        vcl_C = %s(vcl_A, vcl_x)
""" + test_code_footer) % (op[0], layout, dt.__name__, op[1], op[2])
        exec(test_code)


    for op in AB_matrix_operations:
        test_code = (test_code_header + """
    for getter1, getter2 in product(matrix_getters, matrix_getters):
        numpy_A, vcl_A = getter1(size1, size2, layout, dt, None)
        numpy_B, vcl_B = getter2(size1, size2, layout, dt, None)
        numpy_C = %s(numpy_A, numpy_B)
        vcl_C = %s(vcl_A, vcl_B)
""" + test_code_footer) % (op[0], layout, dt.__name__, op[1], op[2])
        exec(test_code)


    for op in ABp_matrix_operations:
        test_code = (test_code_header + """
    for getter1, getter2, getter3 in product(matrix_getters, matrix_getters, scalar_getters):
        numpy_A, vcl_A = getter1(size1, size2, layout, dt, None)
        numpy_B, vcl_B = getter2(size1, size2, layout, dt, None)
        alpha = getter3(dt)
        numpy_C = %s(numpy_A, numpy_B, alpha.value)
        vcl_C = %s(vcl_A, vcl_B, alpha)
""" + test_code_footer) % (op[0], layout, dt.__name__, op[1], op[2])
        exec(test_code)


# TODO
# + initialisers: scalars, ndarray, Matrix
# + non-square matrices, trans matrices
