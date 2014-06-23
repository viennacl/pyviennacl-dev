"""
TODO docstring
"""

from pyviennacl import _viennacl as _v

try:
    vcl = _v.opencl_support
except ImportError:
    raise ImportError("OpenCL support not included in this installation of ViennaCL")

import pyopencl as ocl

vcl_ocl_mapping = {
    vcl.platform: ocl.Platform,
    vcl.device: ocl.Device,
    vcl.context: ocl.Context
}

ocl_vcl_mapping = {o: v for v, o in vcl_ocl_mapping.items()}

def get_pyopencl_object(vcl_object):
    return vcl_ocl_mapping[type(vcl_object)].from_int_ptr(vcl_object.int_ptr)

def get_viennacl_object(ocl_object):
    return ocl_vcl_mapping[type(ocl_object)](ocl_object.int_ptr)

