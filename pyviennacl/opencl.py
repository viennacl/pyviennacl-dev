"""
TODO docstring
"""

import pyviennacl as p
from . import _viennacl as _v

import logging
log = logging.getLogger(__name__)

try:
    vcl = _v.opencl_support
except ImportError:
    raise ImportError("OpenCL support not included in this installation of ViennaCL")

from collections import MutableMapping

import pyopencl as cl
import pyopencl.array

vcl_ocl_type_mapping = {
    vcl.platform: cl.Platform,
    vcl.device: cl.Device,
    vcl.context: cl.Context,
    vcl.command_queue: cl.CommandQueue,
    vcl.program: cl.Program
}

ocl_vcl_type_mapping = {o: v for v, o in vcl_ocl_type_mapping.items()}

vcl_ocl_instance_mapping = {}
ocl_vcl_instance_mapping = {}

context_ids = {}
active_context = [None]

#vcl_backend = vcl.backend()
default_context = cl.Context()

class ContextPrograms(MutableMapping):
    def __init__(self, context):
        self.context = context
        self.vcl_sub_context = context.vcl_sub_context

    def __getitem__(self, index):
        if self.vcl_sub_context.has_program(index):
            program = self.vcl_sub_context.get_program_from_string(index)
            return get_pyopencl_object(program)
        else:
            raise IndexError("Program %s not in context" % index)
    
    def __setitem__(self, index, program):
        if self.vcl_sub_context.has_program(index):
            log.warn("Program %s already in context, so overwriting")
            del self[index]
        self.vcl_sub_context.add_program(program.int_ptr, index)

    def __delitem__(self, index):
        if self.vcl_sub_context.has_program(index):
            self.vcl_sub_context.delete_program(index)
        else:
            raise IndexError("Program %s not in context" % index)

    def _as_list(self):
        programs_list = []
        for index in range(self.vcl_sub_context.program_num):
            program = self.vcl_sub_context.get_program_from_int(index)
            name = program.name
            program = get_pyopencl_object(program)
            programs_list.append((name, program))
        return programs_list

    def keys(self):
        return map(lambda x: x[0], self._as_list())

    def values(self):
        return map(lambda x: x[1], self.as_list())

    def __len__(self):
        return len(self._as_list())

    def __iter__(self):
        return iter(self.keys())

    def __str__(self):
        return dict(self).__str__()

    def __repr__(self):
        return dict(self).__repr__()


def set_active_context(ctx):
    pass
#    if active_context[0] == ctx.sub_context:
#        return
#    if ctx.sub_context in context_ids.keys():
#        vcl.backend.switch_context(context_ids[ctx.sub_context])
#    else:
#        if context_ids:
#            new_id = max(context_ids.values()) + 1
#        else:
#            new_id = 128
#        vcl.backend().add_context(new_id, ctx.vcl_sub_context)
#        vcl.backend().switch_context(new_id)
#        context_ids[ctx.sub_context] = new_id
#    active_context[0] = ctx.sub_context
           
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

    if isinstance(ocl_object, cl.CommandQueue):
        new_vcl_instance = ocl_vcl_type_mapping[type(ocl_object)](
            get_viennacl_object(context), ocl_object.int_ptr)
    else:
        new_vcl_instance = ocl_vcl_type_mapping[type(ocl_object)](ocl_object.int_ptr)
    update_instance_mapping(new_vcl_instance, ocl_object)
    return new_vcl_instance

