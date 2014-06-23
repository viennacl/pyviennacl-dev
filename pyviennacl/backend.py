"""
TODO:
* ViennaCL context object
  + interface with PyOpenCL context
* Construct data types with associated context
* Get associated context from data types
"""
from pyviennacl import _viennacl as _v

WITH_OPENCL = True
try:
    import pyviennacl.opencl as vcl
    import pyopencl as ocl
except ImportError:
    WITH_OPENCL = False

import logging

log = logging.getLogger(__name__)

class MemoryDomain(object):
    pass

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


class Context(object):
    domain = UninitializedMemory
    vcl_context = None
    sub_context = None

    def __init__(self, domain_or_context = DefaultMemory):
        if isinstance(domain_or_context, MemoryDomain):
            self.domain = domain_or_context
            self.vcl_context = _v.context(self.domain.vcl_memory_type)
            return

        if isinstance(domain_or_context, Context):
            self.domain = domain_or_context.domain
            self.vcl_context = domain_or_context.vcl_context

        if WITH_OPENCL:
            if isinstance(domain_or_context, ocl.Context):
                self.domain = OpenCLMemory
                self.sub_context = domain_or_context
                self.vcl_context = _v.context(vcl.get_viennacl_object(domain_or_context))
                return
