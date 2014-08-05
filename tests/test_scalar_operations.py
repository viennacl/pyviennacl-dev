from __future__ import division

import pyviennacl as p
import numpy as np
from _common import *
from itertools import product

lhs_rhs_scalar_getters = [('host_scalar', 'get_host_scalar'),
                          ('device_scalar', 'get_device_scalar')]
rhs_scalar_getters = [('host_scalar', 'get_host_scalar'),
                      ('device_scalar', 'get_device_scalar'),
                      ('scalar_from_vector_norm', 'get_scalar_from_vector_norm'),
                      ('scalar_from_vector_dot_product', 'get_scalar_from_vector_dot_product')]

if double_support:
    dtype_tolerances = [('float32', 1.0E-3), ('float64', 1.0E-11)]
else:
    dtype_tolerances = [('float32', 1.0E-3)]


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
def test_vcl_%s_vcl_%s_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

vcl_numpy_test_code_header = """
def test_vcl_%s_numpy_%s_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

vcl_vcl_vcl_test_code_header = """
def test_vcl_%s_vcl_%s_vcl_%s_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

numpy_vcl_vcl_test_code_header = """
def test_numpy_%s_vcl_%s_vcl_%s_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

vcl_numpy_vcl_test_code_header = """
def test_vcl_%s_numpy_%s_vcl_%s_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

vcl_vcl_numpy_test_code_header = """
def test_vcl_%s_vcl_%s_numpy_%s_%s_%s():
    size = 10
    alpha1 = p.Scalar
"""

test_code_footer = """
    act_diff = math.fabs(diff(numpy_C, vcl_C))
    assert act_diff <= tol, "diff was {} > tolerance {}".format(act_diff, tol)
"""

for d_t_, getter1_, getter2_, op_ in product(dtype_tolerances, lhs_rhs_scalar_getters, rhs_scalar_getters, pq_scalar_operations):
    dt = d_t_[0]
    tol = d_t_[1]

    test_code = (vcl_vcl_test_code_header + """
    alpha = %s(dt)
    beta = %s(dt)
    numpy_C = %s(alpha.value, beta.value)
    vcl_C = %s(alpha, beta)
""" + test_code_footer) % (getter1_[0], getter2_[0], op_[0], dt, getter1_[1], getter2_[1], op_[1], op_[2])
    exec(test_code)

    test_code = (vcl_numpy_test_code_header + """
    alpha = %s(dt)
    beta = %s(dt)
    numpy_C = %s(alpha.value, beta.value)
    vcl_C = %s(alpha, beta.value)
""" + test_code_footer) % (getter1_[0], getter2_[0], op_[0], dt, getter1_[1], getter2_[1], op_[1], op_[2])
    exec(test_code)

for d_t_, getter1_, getter2_, getter3_, op_ in product(dtype_tolerances, lhs_rhs_scalar_getters, rhs_scalar_getters, rhs_scalar_getters, pqr_scalar_operations):
    dt = d_t_[0]
    tol = d_t_[1]
    test_code = (vcl_vcl_vcl_test_code_header + """
    alpha = %s(dt)
    beta = %s(dt)
    gamma = %s(dt)
    numpy_C = %s(alpha.value, beta.value, gamma.value)
    vcl_C = %s(alpha, beta, gamma)
""" + test_code_footer) % (getter1_[0], getter2_[0], getter3_[0], op_[0], dt, getter1_[1], getter2_[1], getter3_[1], op_[1], op_[2])
    exec(test_code)

    test_code = (numpy_vcl_vcl_test_code_header + """
    alpha = %s(dt)
    beta = %s(dt)
    gamma = %s(dt)
    numpy_C = %s(alpha.value, beta.value, gamma.value)
    vcl_C = %s(alpha.value, beta, gamma)
""" + test_code_footer) % (getter1_[0], getter2_[0], getter3_[0], op_[0], dt, getter1_[1], getter2_[1], getter3_[1], op_[1], op_[2])
    exec(test_code)

    test_code = (vcl_numpy_vcl_test_code_header + """
    alpha = %s(dt)
    beta = %s(dt)
    gamma = %s(dt)
    numpy_C = %s(alpha.value, beta.value, gamma.value)
    vcl_C = %s(alpha, beta.value, gamma)
""" + test_code_footer) % (getter1_[0], getter2_[0], getter3_[0], op_[0], dt, getter1_[1], getter2_[1], getter3_[1], op_[1], op_[2])
    exec(test_code)

    test_code = (vcl_vcl_numpy_test_code_header + """
    alpha = %s(dt)
    beta = %s(dt)
    gamma = %s(dt)
    numpy_C = %s(alpha.value, beta.value, gamma.value)
    vcl_C = %s(alpha, beta, gamma.value)
""" + test_code_footer) % (getter1_[0], getter2_[0], getter3_[0], op_[0], dt, getter1_[1], getter2_[1], getter3_[1], op_[1], op_[2])
    exec(test_code)
