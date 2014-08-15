"""Memory and compute control"""

import os
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
    """
    TODO docstring
    """
    vcl_sub_context = None

    def __init__(self, *args, **kwargs):
        """
        TODO docstring
        """
        if len(args) == 1:
            if WITH_OPENCL:
                if isinstance(args[0], cl.MemoryObject):
                    int_ptr = args[0].int_ptr
                    raw_size = args[0].size
                    self.vcl_handle = _v.mem_handle()
                    self.vcl_handle.init_opencl_handle(int_ptr, raw_size)
                    return

            try:
                self.vcl_handle = args[0].vcl_handle
            except AttributeError:
                self.vcl_handle = args[0]

        elif len(args) == 3:
            domain = args[0]
            int_ptr = args[1]
            raw_size = args[2]
            vcl_handle = _v.mem_handle()
            if domain is MainMemory:
                vcl_handle.init_ram_handle(int_ptr, raw_size)
            elif domain is OpenCLMemory:
                vcl_handle.init_opencl_handle(int_ptr, raw_size)
            elif domain is CUDAMemory:
                vcl_handle.init_cuda_handle(int_ptr, raw_size)
            else:
                raise MemoryError("Domain not understood")
            self.vcl_handle = vcl_handle

        else:
            raise TypeError("Cannot construct MemoryHandle like this")

    def __eq__(self, other):
        return self.vcl_handle == other.vcl_handle

    def __ne__(self, other):
        return self.vcl_handle != other.vcl_handle

    def __lt__(self, other):
        return self.vcl_handle < other.vcl_handle

    def __gt__(self, other):
        return other.vcl_handle < self.vcl_handle

    def assign(self, other):
        if other.domain is MainMemory:
            self.vcl_handle.init_ram_handle(other.int_ptr, other.raw_size)
        elif other.domain is OpenCLMemory:
            self.vcl_handle.init_opencl_handle(other.int_ptr, other.raw_size)
        elif other.domain is CUDAMemory:
            self.vcl_handle.init_cuda_handle(other.int_ptr, other.raw_size)
        else:
            raise MemoryError("Other memory domain not understood")

    def swap(self, other):
        self.vcl_handle.swap(other.vcl_handle)
        tmp_vcl_handle = self.vcl_handle
        self.__init__(other.vcl_handle)
        other.__init__(tmp_vcl_handle)

    @property
    def domain(self):
        return vcl_memory_types[self.vcl_handle.active_handle_id]

    @domain.setter
    def domain(self, domain):
        self.vcl_handle.active_handle_id = domain.vcl_memory_type

    @property
    def context(self):
        if self.domain is not OpenCLMemory:
            return Context(self.domain)

        if self.vcl_sub_context:
            return Context(self.vcl_sub_context)
        else:
            return None

    @context.setter
    def context(self, ctx):
        if self.domain is not ctx.domain:
            raise MemoryError("Context memory domains not identical")
        if self.domain is OpenCLMemory:
            self.vcl_handle.opencl_context = ctx.vcl_sub_context

    @property
    def raw_size(self):
        return self.vcl_handle.raw_size

    @raw_size.setter
    def raw_size(self, size):
        self.vcl_handle.raw_size = size

    @property
    def int_ptr(self):
        if self.domain is UninitializedMemory:
            return 0
        elif self.domain is MainMemory:
            return self.vcl_handle.ram_handle
        elif self.domain is OpenCLMemory:
            return self.vcl_handle.opencl_handle
        elif self.domain is CUDAMemory:
            return self.vcl_handle.cuda_handle

    @int_ptr.setter
    def int_ptr(self, ptr):
        if self.domain is MainMemory:
            self.vcl_handle.ram_handle = int_ptr
        elif domain is OpenCLMemory:
            self.vcl_handle.opencl_handle = int_ptr
        elif domain is CUDAMemory:
            self.vcl_handle.cuda_handle = int_ptr
        else:
            raise MemoryError("Domain not understood")

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
                new_path = appdirs.user_data_dir
                if not os.path.isdir(new_path):
                    try: os.makedirs(new_path)
                    except: pass
                try:
                    new_path = os.path.join(new_path, '')
                    open(os.path.join(new_path, 'permission_test'), 'a+')
                except OSError as e:
                    log.warning("Could not open cache path '%s' for writing, disabling kernel cache. Exception was: %s" % (new_path, e))
                    new_path = ''
                self.cache_path = new_path
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
        return vcl.ContextQueues(self)

    def add_queue(self, device, queue = None):
        """
        TODO docstring
        """
        if self.domain is not OpenCLMemory:
            raise TypeError("Only the OpenCL backend currently supports queues")
        self.queues[device].append(queue)

    def switch_queue(self, queue):
        """
        TODO docstring
        """
        if self.domain is not OpenCLMemory:
            raise TypeError("Only the OpenCL backend currently supports queues")
        self.queues.switch_queue(queue)

    @property
    def current_queue(self):
        if self.domain is not OpenCLMemory:
            raise TypeError("Only the OpenCL backend currently supports multiple devices")
        return self.queues.current_queue

    def finish_all_queues(self):
        """
        TODO docstring
        """
        if self.domain is not OpenCLMemory:
            raise TypeError("This only makes sense on OpenCL")
        for device in self.devices:
            self.queues[device].finish()

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


__all__ = ['UninitializedMemory', 'MainMemory', 'OpenCLMemory', 'CUDAMemory',
           'DefaultMemory', 'MemoryHandle', 'Context']
