"""
TODO docstring
"""

import pyviennacl as p
from . import _viennacl as _v

try:
    vcl = _v.opencl_support
except ImportError:
    raise ImportError("OpenCL support not included in this installation of ViennaCL")

import pyopencl as cl
import pyopencl.array

vcl_ocl_type_mapping = {
    vcl.platform: cl.Platform,
    vcl.device: cl.Device,
    vcl.context: cl.Context,
    vcl.command_queue: cl.CommandQueue
}

ocl_vcl_type_mapping = {o: v for v, o in vcl_ocl_type_mapping.items()}

vcl_ocl_instance_mapping = {}
ocl_vcl_instance_mapping = {}

context_ids = {}
active_context = [None]

def set_active_context(ctx):
    return # This is now a no-op
    #if active_context[0] == ctx.sub_context:
    #    return
    #if ctx.sub_context in context_ids.keys():
    #    vcl.backend.switch_context(context_ids[ctx.sub_context])
    #else:
    #    if context_ids:
    #        new_id = max(context_ids.values()) + 1
    #    else:
    #        new_id = 128
    #    print("!!!!!!!! new_id", new_id)
    #    vcl.backend.add_context(new_id, ctx.vcl_sub_context)
    #    vcl.backend.switch_context(new_id)
    #    context_ids[ctx.sub_context] = new_id
    #active_context[0] = ctx.sub_context
            
def update_instance_mapping(vcl_object, ocl_object):
    vcl_ocl_instance_mapping[vcl_object.int_ptr] = ocl_object
    ocl_vcl_instance_mapping[ocl_object.int_ptr] = vcl_object    

def get_pyopencl_object(vcl_object):
    if isinstance(vcl_object, p.MagicMethods):
        return vcl_object.as_opencl_array()

    if vcl_object.int_ptr in vcl_ocl_instance_mapping.keys():
        return vcl_ocl_instance_mapping[vcl_object.int_ptr]

    new_ocl_instance = vcl_ocl_type_mapping[type(vcl_object)].from_int_ptr(vcl_object.int_ptr)
    update_instance_mapping(vcl_object, new_ocl_instance)
    return new_ocl_instance

def get_viennacl_object(ocl_object, context = None):
    if isinstance(ocl_object, cl.array.Array):
        raise TypeError("TODO: Create VCL object from Array")

    if isinstance(ocl_object, cl.MemoryObject):
        raise TypeError("Unsure how to proceed: please create the ViennaCL object manually")

    if ocl_object.int_ptr in ocl_vcl_instance_mapping.keys():
        return ocl_vcl_instance_mapping[ocl_object.int_ptr]

    if isinstance(ocl_object, ocl.CommandQueue):
        new_vcl_instance = ocl_vcl_type_mapping[type(ocl_object)](
            get_viennacl_object(context), ocl_object.int_ptr)
    else:
        new_vcl_instance = ocl_vcl_type_mapping[type(ocl_object)](ocl_object.int_ptr)
    update_instance_mapping(new_vcl_instance, ocl_object)
    return new_vcl_instance

