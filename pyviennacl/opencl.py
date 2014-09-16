"""
TODO docstring
"""

import pyviennacl as p
from . import _viennacl as _v

import logging
log = logging.getLogger(__name__)

try:
    vcl = _v.opencl_support
except AttributeError:
    raise ImportError("OpenCL support not included in this installation of ViennaCL")

from collections import MutableMapping, MutableSequence

import pyopencl as cl
import pyopencl.array

VendorId = _v.opencl_support.vendor_id;

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
default_context = cl.create_some_context(interactive=False)


class ContextDict(MutableMapping):
    def __init__(self, context):
        self.context = context
        self.sub_context = context.sub_context
        self.vcl_sub_context = context.vcl_sub_context

    def __iter__(self):
        return iter(self.keys())

    def __str__(self):
        return dict(self).__str__()

    def __repr__(self):
        return dict(self).__repr__()


class ContextPrograms(ContextDict):
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


class DeviceQueues(MutableSequence):
    def __init__(self, context, device):
        self.context = context
        self.device = device

    @property
    def device_queues(self):
        device_queues = []
        idx = 0
        while True:
            try:
                vcl_queue = self.context.vcl_sub_context.get_queue(self.device.int_ptr, idx)
            except RuntimeError:
                break
            queue = get_pyopencl_object(vcl_queue)
            device_queues.append(queue)
            idx += 1
        return device_queues

    def __getitem__(self, index):
        return self.device_queues[index]

    def __len__(self):
        return len(self.device_queues)

    def __repr__(self):
        return repr(self.device_queues)

    def __str__(self):
        return str(self.device_queues)

    def __getitem__(self, index):
        return list(self)[index]

    def __setitem__(self):
        raise TypeError("Use append instead; the queue order is not mutable!")

    def __delitem__(self):
        raise TypeError("Cannot delete queues from context")

    def insert(self, index, item):
        raise TypeError("Use append instead; the queue order is not mutable!")

    def append(self, queue):
        if queue is None:
            queue = cl.CommandQueue(self.context.sub_context, self.device)
        if queue in self.device_queues:
            log.warn("Queue %s already in context for device %s; doing nothing" % (queue, device))
            return
        self.context.vcl_sub_context.add_existing_queue(self.device.int_ptr,
                                                        queue.int_ptr)

    def finish(self):
        for queue in self.device_queues:
            queue.finish()


class ContextQueues(ContextDict):
    def _get_queues_dict(self):
        queues = {}
        for d in self.context.devices:
            queues[d] = DeviceQueues(self.context, d)
        return queues

    def __getitem__(self, device):
        try: return DeviceQueues(self.context, device)
        except: IndexError("Device %s not in context" % device)

    def __setitem__(self, device, queues):
        for queue in queues:
            self[device].append(queue)

    def __delitem__(self, index):
        raise NotImplementedError("Cannot delete queues from context")

    def keys(self):
        return self.context.devices

    def values(self):
        return self._get_queues_dict().values()

    def __len__(self):
        return len(self.values())

    @property
    def current_queue(self):
        return get_pyopencl_object(self.vcl_sub_context.current_queue)

    def switch_queue(self, queue):
        if queue not in self[queue.device]:
            self[queue.device].append(queue)
        vcl_queue = get_viennacl_object(queue, self.sub_context)
        self.vcl_sub_context.switch_queue(vcl_queue)



def architecture_family(vendor_id, name):
    return str(_v.opencl_support.get_architecture_family(vendor_id, name))

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


__all__ = ['ContextPrograms', 'DeviceQueues', 'ContextQueues',
           'architecture_family', 'get_pyopencl_object', 'get_viennacl_object']
