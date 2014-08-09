"""
TODO:
* ViennaCL context object
  + interface with PyOpenCL context
* Construct data types with associated context
* Get associated context from data types
"""
import pyviennacl as p
from . import _viennacl as _v

from appdirs import AppDirs
appdirs = AppDirs("pyviennacl", "viennacl", version=p.__version__)

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
    _programs = None

    def __init__(self, domain_or_context=DefaultMemory):
        if domain_or_context is None:
            domain_or_context = DefaultMemory

        if isinstance(domain_or_context, Context):
            self.domain = domain_or_context.domain
            self.vcl_context = domain_or_context.vcl_context
            self.sub_context = domain_or_context.sub_context
            #if self.domain is OpenCLMemory:
            #    vcl.set_active_context(self)
            return

        if WITH_OPENCL:
            if isinstance(domain_or_context, cl.Context):
                self.domain = OpenCLMemory
                self.sub_context = domain_or_context
                create_vcl_context_from = vcl.get_viennacl_object(self.sub_context)

        try:
            if issubclass(domain_or_context, MemoryDomain): # cf default arg
                self.domain = domain_or_context
                if domain_or_context is OpenCLMemory:
                    self.sub_context = vcl.default_context
                    create_vcl_context_from = vcl.get_viennacl_object(self.sub_context)
                else:
                    create_vcl_context_from = self.domain.vcl_memory_type
        except TypeError: pass

        self.vcl_context = _v.context(create_vcl_context_from)
        if self.domain is OpenCLMemory:
            for device in self.devices:
                if not self.queues[device]:
                    self.add_queue(device)
            if not self.cache_path:
                self.cache_path = appdirs.user_config_dir
        #if self.domain is OpenCLMemory:
        #    vcl.set_active_context(self)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self.domain != other.domain:
            return False
        if self.domain is OpenCLMemory:
            if self.vcl_sub_context.int_ptr != other.vcl_sub_context.int_ptr:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    @property
    def cache_path(self):
        if self.domain is not OpenCLMemory:
            raise TypeError("Only OpenCL contexts support kernel caching")
        try: return self.vcl_sub_context.cache_path
        except: return ''

    @cache_path.setter
    def cache_path(self, path):
        if self.domain is not OpenCLMemory:
            raise TypeError("Only OpenCL contexts support kernel caching")
        self.vcl_sub_context.cache_path = path

    @property
    def vcl_sub_context(self):
        if self.domain is not OpenCLMemory:
            raise TypeError("Only OpenCL sub-context supported currently")
        return vcl.get_viennacl_object(self.sub_context, self)

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

    def finish_all_queues(self):
        """
        TODO docstring
        """
        if self.domain is not OpenCLMemory:
            raise TypeError("This only makes sense on OpenCL")
        for device in self.queues.keys():
            for queue in self.queues[device]:
                queue.finish()

    @property
    def programs(self):
        if self.domain is not OpenCLMemory:
            raise TypeError("This only makes sense on OpenCL")
        if not isinstance(self._programs, vcl.ContextPrograms):
            self._programs = vcl.ContextPrograms(self)
        return self._programs

    @property
    def default_dtype(self):
        if self.domain is not OpenCLMemory:
            return p.dtype(p.float64)
        if self.current_device.double_fp_config:
            return p.dtype(p.float64)
        else:
            return p.dtype(p.float32)


def backend_finish():
    """
    Block until any computation active on the compute backend is finished.
    """
    return _v.backend_finish()

