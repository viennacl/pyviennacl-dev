from __future__ import division

import pyviennacl as p
import numpy as np
from _common import *
from itertools import product

lhs_rhs_scalar_getters = [get_host_scalar, get_device_scalar]
rhs_scalar_getters = [get_host_scalar, get_device_scalar, get_scalar_from_vector_norm, get_scalar_from_vector_dot_product]
dtype_tolerances = [(p.float32, 1.0E-3), (p.float64, 1.0E-11)]


p_scalar_operations = [
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

pq_scalar_operations = [
    ('add', 'add', 'add'),
    ('iadd', 'iadd', 'iadd'),
    ('sub', 'sub', 'sub'),
    ('isub', 'isub', 'isub'),
    ('mul', 'mul', 'mul'),
    ('imul', 'imul', 'imul'),
    ('div', 'div', 'div'),
    ('idiv', 'idiv', 'idiv'),
    ('pow', 'pow', 'pow'),
    ('ipow', 'ipow', 'ipow')
 ]

pqr_scalar_operations = [
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


vcl_vcl_test_code_header = """
def test_vcl_vcl_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

vcl_vcl_vcl_test_code_header = """
def test_vcl_vcl_vcl_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

numpy_vcl_vcl_test_code_header = """
def test_numpy_vcl_vcl_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

vcl_numpy_vcl_test_code_header = """
def test_vcl_numpy_vcl_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

vcl_vcl_numpy_test_code_header = """
def test_vcl_vcl_numpy_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

vcl_numpy_test_code_header = """
def test_vcl_numpy_%s_%s():
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

    for op in pq_scalar_operations:
        test_code = (vcl_vcl_test_code_header + """
    for getter1, getter2 in product(lhs_rhs_scalar_getters, rhs_scalar_getters):
        alpha = getter1(dt)
        beta = getter2(dt)
        numpy_C = %s(alpha.value, beta.value)
        vcl_C = %s(alpha, beta)
""" + test_code_footer) % (op[0], dt.__name__, op[1], op[2])
        exec(test_code)

    for op in pq_scalar_operations:
        test_code = (vcl_numpy_test_code_header + """
    for getter1, getter2 in product(lhs_rhs_scalar_getters, rhs_scalar_getters):
        alpha = getter1(dt)
        beta = getter2(dt)
        numpy_C = %s(alpha.value, beta.value)
        vcl_C = %s(alpha, beta.value)
""" + test_code_footer) % (op[0], dt.__name__, op[1], op[2])
        exec(test_code)

    for op in pqr_scalar_operations:
        test_code = (vcl_vcl_vcl_test_code_header + """
    for getter1, getter2, getter3 in product(lhs_rhs_scalar_getters, rhs_scalar_getters, rhs_scalar_getters):
        alpha = getter1(dt)
        beta = getter2(dt)
        gamma = getter3(dt)
        numpy_C = %s(alpha.value, beta.value, gamma.value)
        vcl_C = %s(alpha, beta, gamma)
""" + test_code_footer) % (op[0], dt.__name__, op[1], op[2])
        exec(test_code)

    for op in pqr_scalar_operations:
        test_code = (numpy_vcl_vcl_test_code_header + """
    for getter1, getter2, getter3 in product(lhs_rhs_scalar_getters, rhs_scalar_getters, rhs_scalar_getters):
        alpha = getter1(dt)
        beta = getter2(dt)
        gamma = getter3(dt)
        numpy_C = %s(alpha.value, beta.value, gamma.value)
        vcl_C = %s(alpha.value, beta, gamma)
""" + test_code_footer) % (op[0], dt.__name__, op[1], op[2])
        exec(test_code)

    for op in pqr_scalar_operations:
        test_code = (vcl_numpy_vcl_test_code_header + """
    for getter1, getter2, getter3 in product(lhs_rhs_scalar_getters, rhs_scalar_getters, rhs_scalar_getters):
        alpha = getter1(dt)
        beta = getter2(dt)
        gamma = getter3(dt)
        numpy_C = %s(alpha.value, beta.value, gamma.value)
        vcl_C = %s(alpha, beta.value, gamma)
""" + test_code_footer) % (op[0], dt.__name__, op[1], op[2])
        exec(test_code)

    for op in pqr_scalar_operations:
        test_code = (vcl_vcl_numpy_test_code_header + """
    for getter1, getter2, getter3 in product(lhs_rhs_scalar_getters, rhs_scalar_getters, rhs_scalar_getters):
        alpha = getter1(dt)
        beta = getter2(dt)
        gamma = getter3(dt)
        numpy_C = %s(alpha.value, beta.value, gamma.value)
        vcl_C = %s(alpha, beta, gamma.value)
""" + test_code_footer) % (op[0], dt.__name__, op[1], op[2])
        exec(test_code)
