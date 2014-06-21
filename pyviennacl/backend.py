"""
TODO:
* ViennaCL context object
  + interface with PyOpenCL context
* Construct data types with associated context
* Get associated context from data types
"""
from pyviennacl import _viennacl as _v
import logging

log = logging.getLogger(__name__)

class MemoryType(object):
    pass

class UninitializedMemory(MemoryType):
    vcl_memory_type = _v.memory_types.MEMORY_NOT_INITIALIZED

class MainMemory(MemoryType):
    vcl_memory_type = _v.memory_types.MAIN_MEMORY

class OpenCLMemory(MemoryType):
    vcl_memory_type = _v.memory_types.OPENCL_MEMORY

class CUDAMemory(MemoryType):
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

    def __init__(self, domain = DefaultMemory):
        self.domain = domain
        self.vcl_context = _v.context(domain.vcl_memory_type)
