from __future__ import division

import pyviennacl as p
import numpy as np
from _common import *
from itertools import product

vector_getters = [get_vector, get_vector_range, get_vector_slice]
scalar_getters = [get_host_scalar, get_device_scalar]
dtype_tolerances = [(p.float32, 1.0E-3), (p.float64, 1.0E-11)]


x_vector_operations = [
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


test_code_header = """
def test_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

test_code_footer = """
        act_diff = math.fabs(diff(numpy_C, vcl_C))
        assert act_diff <= tol, "diff was {} > tolerance {}".format(act_diff, tol)
"""

for d_t in dtype_tolerances:
    dt = d_t[0]
    tol = d_t[1]

    for op in x_vector_operations:
        test_code = (test_code_header + """
    for getter1 in vector_getters:
        numpy_A, vcl_A = getter1(size, dt)
        numpy_C = %s(numpy_A)
        vcl_C = %s(vcl_A)
""" + test_code_footer) % (op[0], dt.__name__, op[1], op[2])
        exec(test_code)

    for op in xp_vector_operations:
        test_code = (test_code_header + """
    for getter1, getter2 in product(vector_getters, scalar_getters):
        numpy_A, vcl_A = getter1(size, dt)
        alpha = getter2(dt)
        numpy_C = %s(numpy_A, alpha.value)
        vcl_C = %s(vcl_A, alpha)
""" + test_code_footer) % (op[0], dt.__name__, op[1], op[2])
        exec(test_code)

    for op in xy_vector_operations:
        test_code = (test_code_header + """
    for getter1, getter2 in product(vector_getters, vector_getters):
        numpy_A, vcl_A = getter1(size, dt)
        numpy_B, vcl_B = getter2(size, dt)
        numpy_C = %s(numpy_A, numpy_B)
        vcl_C = %s(vcl_A, vcl_B)
""" + test_code_footer) % (op[0], dt.__name__, op[1], op[2])
        exec(test_code)


    for op in xyp_vector_operations:
        test_code = (test_code_header + """
    for getter1, getter2, getter3 in product(vector_getters, vector_getters, scalar_getters):
        numpy_A, vcl_A = getter1(size, dt)
        numpy_B, vcl_B = getter2(size, dt)
        alpha = getter3(dt)
        numpy_C = %s(numpy_A, numpy_B, alpha.value)
        vcl_C = %s(vcl_A, vcl_B, alpha)
""" + test_code_footer) % (op[0], dt.__name__, op[1], op[2])
        exec(test_code)


# TODO
# + initialisers: scalars, ndarray
# + negation
# + unary operations
# + in-place multiply-division-add
# + plane rotation

