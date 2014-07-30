from __future__ import division

import pyviennacl as p
import numpy as np
from _common import *
from itertools import product

vector_getters = [('vector', 'get_vector'),
                  ('vector_range', 'get_vector_range'),
                  ('vector_slice', 'get_vector_slice')]
scalar_getters = [('host_scalar', 'get_host_scalar'),
                  ('device_scalar', 'get_device_scalar')]
dtype_tolerances = [(p.float32, 1.0E-3), (p.float64, 1.0E-11)]


x_vector_operations = [
#    ('abs', 'np.absolute', 'p.absolute'),
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

xp_vector_operations = [
    ('mul', 'mul', 'mul'),
    ('div', 'div', 'div')
]

xy_vector_operations = [
    ('add', 'add', 'add'),
    ('iadd', 'iadd', 'iadd'),
    ('sub', 'sub', 'sub'),
    ('isub', 'isub', 'isub'),
    ('dot_vector', 'dot', 'dot'),

    #('elementwise_pow', 'pow', 'pow'),
    ('elementwise_prod', 'mul', 'p.ElementProd'),
    ('elementwise_div', 'div', 'p.ElementDiv')
 ]

xyp_vector_operations = [
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


x_test_code_header = """
def test_%s_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

xp_test_code_header = """
def test_%s_%s_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""
xy_test_code_header = xp_test_code_header

xyp_test_code_header = """
def test_%s_%s_%s_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

test_code_footer = """
    act_diff = math.fabs(diff(numpy_C, vcl_C))
    assert act_diff <= tol, "diff was {} > tolerance {}".format(act_diff, tol)
"""

for d_t, getter1_, op_ in product(dtype_tolerances, vector_getters, x_vector_operations):
    dt = d_t[0]
    tol = d_t[1]

    test_code = (x_test_code_header + """
    numpy_A, vcl_A = %s(size, dt)
    numpy_C = %s(numpy_A)
    vcl_C = %s(vcl_A)
""" + test_code_footer) % (getter1_[0], op_[0], dt.__name__, getter1_[1], op_[1], op_[2])
    exec(test_code)

for d_t, getter1_, getter2_, op_ in product(dtype_tolerances, vector_getters, scalar_getters, xp_vector_operations):
    dt = d_t[0]
    tol = d_t[1]
    test_code = (xp_test_code_header + """
    numpy_A, vcl_A = %s(size, dt)
    alpha = %s(dt)
    numpy_C = %s(numpy_A, alpha.value)
    vcl_C = %s(vcl_A, alpha)
""" + test_code_footer) % (getter1_[0], getter2_[0], op_[0], dt.__name__, getter1_[1], getter2_[1], op_[1], op_[2])
    exec(test_code)

for d_t, getter1_, getter2_, op_ in product(dtype_tolerances, vector_getters, vector_getters, xy_vector_operations):
    dt = d_t[0]
    tol = d_t[1]
    test_code = (xy_test_code_header + """
    numpy_A, vcl_A = %s(size, dt)
    numpy_B, vcl_B = %s(size, dt)
    numpy_C = %s(numpy_A, numpy_B)
    vcl_C = %s(vcl_A, vcl_B)
""" + test_code_footer) % (getter1_[0], getter2_[0], op_[0], dt.__name__, getter1_[1], getter2_[1], op_[1], op_[2])
    exec(test_code)

for d_t, getter1_, getter2_, getter3_, op_ in product(dtype_tolerances, vector_getters, vector_getters, scalar_getters, xyp_vector_operations):
    dt = d_t[0]
    tol = d_t[1]
    test_code = (xyp_test_code_header + """
    numpy_A, vcl_A = %s(size, dt)
    numpy_B, vcl_B = %s(size, dt)
    alpha = %s(dt)
    numpy_C = %s(numpy_A, numpy_B, alpha.value)
    vcl_C = %s(vcl_A, vcl_B, alpha)
""" + test_code_footer) % (getter1_[0], getter2_[0], getter3_[0], op_[0], dt.__name__, getter1_[1], getter2_[1], getter3_[1], op_[1], op_[2])
    exec(test_code)


# TODO
# + negation
# + in-place multiply-division-add
# + plane rotation

