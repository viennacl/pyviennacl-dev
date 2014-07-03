"""
TODO:
* ViennaCL context object
  + interface with PyOpenCL context
* Construct data types with associated context
* Get associated context from data types
"""
from . import _viennacl as _v

import logging
log = logging.getLogger(__name__)

WITH_OPENCL = True
try:
    import pyviennacl.opencl as vcl
    import pyopencl as cl
except ImportError as e:
    #log.warning("OpenCL not available: %s", e)
    WITH_OPENCL = False

class MemoryDomain(object):
    def __init__(*args, **kwargs):
        raise TypeError("This class is not supposed to be instantiated")

class UninitializedMemory(MemoryDomain):
    vcl_memory_type = _v.memory_types.MEMORY_NOT_INITIALIZED

class MainMemory(MemoryDomain):
    vcl_memory_type = _v.memory_types.MAIN_MEMORY

class OpenCLMemory(MemoryDomain):
    vcl_memory_type = _v.memory_types.OPENCL_MEMORY

class CUDAMemory(MemoryDomain):
    vcl_memory_type = _v.memory_types.CUDA_MEMORY

vcl_memory_types = {
    _v.memory_types.MEMORY_NOT_INITIALIZED: UninitializedMemory,
    _v.memory_types.MAIN_MEMORY: MainMemory,
    _v.memory_types.OPENCL_MEMORY: OpenCLMemory,
    _v.memory_types.CUDA_MEMORY: CUDAMemory
}

DefaultMemory = vcl_memory_types[_v.default_memory_type]


class MemoryHandle(object):
    vcl_handle = None
    domain = UninitializedMemory
    int_ptr = None

    def __init__(self, vcl_handle):
        self.vcl_handle = vcl_handle
        self.domain = vcl_memory_types[vcl_handle.active_handle_id]
        if self.domain is UninitializedMemory:
            self.int_ptr = None
        elif self.domain is MainMemory:
            self.int_ptr = vcl_handle.ram_handle
        elif self.domain is OpenCLMemory:
            self.int_ptr = vcl_handle.opencl_handle
        elif self.domain is CUDAMemory:
            self.int_ptr = vcl_handle.cuda_handle

    def __eq__(self, other):
        return self.vcl_handle == other.vcl_handle

    def __ne__(self, other):
        return self.vcl_handle != other.vcl_handle

    def __lt__(self, other):
        return self.vcl_handle < other.vcl_handle

    def __gt__(self, other):
        return other.vcl_handle < self.vcl_handle

    def swap(self, other):
        self.vcl_handle.swap(other.vcl_handle)
        tmp_vcl_handle = self.vcl_handle
        self.__init__(other.vcl_handle)
        other.__init__(tmp_vcl_handle)

    @property
    def buffer(self):
        if self.domain is not OpenCLMemory:
            raise TypeError("You can currently only get a buffer with the OpenCL backend")
        return cl.MemoryObject.from_int_ptr(self.int_ptr)


class Context(object):
    domain = UninitializedMemory
    vcl_context = None
    sub_context = None
    vcl_sub_context = None

    def __init__(self, domain_or_context = DefaultMemory):
        try:
            if issubclass(domain_or_context, MemoryDomain):
                self.domain = domain_or_context
                self.vcl_context = _v.context(self.domain.vcl_memory_type)
                if domain_or_context is OpenCLMemory:
                    self.vcl_sub_context = self.vcl_context.opencl_context
                    self.sub_context = vcl.get_pyopencl_object(self.vcl_sub_context)                    
                return
        except TypeError: pass

        if isinstance(domain_or_context, Context):
            self.domain = domain_or_context.domain
            self.vcl_context = domain_or_context.vcl_context
            self.sub_context = domain_or_context.sub_context
            self.vcl_sub_context = domain_or_context.vcl_sub_context
            return

        if WITH_OPENCL:
            if isinstance(domain_or_context, cl.Context):
                self.domain = OpenCLMemory
                self.sub_context = domain_or_context
                self.vcl_sub_context = vcl.get_viennacl_object(domain_or_context)
                self.vcl_context = _v.context(self.vcl_sub_context)
                vcl.set_active_context(self)
                return

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.domain != other.domain:
            return False
        if self.vcl_sub_context != other.vcl_sub_context:
            return False
        return True

    @property
    def opencl_context(self):
        if self.domain is not OpenCLMemory:
            raise TypeError("You can only get the OpenCL context with the OpenCL backend")
        return self.sub_context

    @property
    def devices(self):
        if self.domain is not OpenCLMemory:
            raise TypeError("Only the OpenCL backend currently supports multiple devices")
        return self.sub_context.devices
  
    @property
    def current_device(self):
        if self.domain is not OpenCLMemory:
            raise TypeError("Only the OpenCL backend currently supports multiple devices")
        return vcl.get_pyopencl_object(self.vcl_sub_context.current_device)

    @property
    def queues(self):
        """
        TODO docstring
        """
        if self.domain is not OpenCLMemory:
            raise TypeError("Only the OpenCL backend currently supports multiple devices")
        queues = {}
        for d in self.devices:
            device_queues = []
            idx = 0
            while True:
                try:
                    vcl_queue = self.vcl_sub_context.get_queue(d.int_ptr, idx)
                except RuntimeError:
                    break
                queue = vcl.get_pyopencl_object(vcl_queue)
                device_queues.append(queue)
                idx += 1
            queues[d] = device_queues
        return queues

    def add_queue(self, device, queue = None):
        """
        TODO docstring
        """
        if self.domain is not OpenCLMemory:
            raise TypeError("Only the OpenCL backend currently supports queues")
        if queue is None:
            queue = cl.CommandQueue(self.sub_context, device)
        if queue in self.queues[device]:
            return
        self.vcl_sub_context.add_existing_queue(device.int_ptr, queue.int_ptr)

    def switch_queue(self, queue):
        """
        TODO docstring
        """
        if self.domain is not OpenCLMemory:
            raise TypeError("Only the OpenCL backend currently supports queues")
        if queue not in self.queues[queue.device]:
            self.add_queue(queue.device, queue)
        vcl_queue = vcl.get_viennacl_object(queue, self.sub_context)
        self.vcl_sub_context.switch_queue(vcl_queue)

    @property
    def current_queue(self):
        if self.domain is not OpenCLMemory:
            raise TypeError("Only the OpenCL backend currently supports multiple devices")
        return vcl.get_pyopencl_object(self.vcl_sub_context.current_queue)
