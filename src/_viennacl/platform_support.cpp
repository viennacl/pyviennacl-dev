#include "common.hpp"

#include <viennacl/context.hpp>
#include <viennacl/backend/mem_handle.hpp>

vcl::vcl_size_t get_ram_handle(vcl::backend::mem_handle& m) {
  return (vcl::vcl_size_t) m.ram_handle().get();
}

void set_ram_handle(vcl::backend::mem_handle& m, vcl::vcl_size_t ptr) {
  char* ram_handle_ = (char*) ptr;
  vcl::backend::mem_handle::ram_handle_type ram_handle(ram_handle_);
  m.ram_handle() = ram_handle;
}

void init_ram_handle(vcl::backend::mem_handle& m,
                     vcl::vcl_size_t ptr, vcl::vcl_size_t raw_size) {
  set_ram_handle(m, ptr);
  m.raw_size(raw_size);
  m.switch_active_handle_id(vcl::MAIN_MEMORY);
}

#ifdef VIENNACL_WITH_OPENCL
vcl::vcl_size_t get_opencl_handle(vcl::backend::mem_handle& m) {
  return (vcl::vcl_size_t) m.opencl_handle().get();
}

void set_opencl_handle(vcl::backend::mem_handle& m, vcl::vcl_size_t ptr) {
  cl_mem opencl_handle = (cl_mem) ptr;
  m.opencl_handle() = opencl_handle;
}

void init_opencl_handle(vcl::backend::mem_handle& m,
                        vcl::vcl_size_t ptr, vcl::vcl_size_t raw_size) {
  set_opencl_handle(m, ptr);
  m.raw_size(raw_size);
  m.switch_active_handle_id(vcl::OPENCL_MEMORY);
}

const vcl::ocl::context& get_opencl_handle_context(vcl::backend::mem_handle& m) {
  return m.opencl_handle().context();
}

void set_opencl_handle_context(vcl::backend::mem_handle& m,
                               const vcl::ocl::context& c) {
  m.opencl_handle().context(c);
}
#endif

#ifdef VIENNACL_WITH_CUDA
vcl::vcl_size_t get_cuda_handle(vcl::backend::mem_handle& m) {
  return (vcl::vcl_size_t) m.cuda_handle().get();
}

void set_cuda_handle(vcl::backend::mem_handle& m, vcl::vcl_size_t ptr) {
  char* cuda_handle_ = (char*) ptr;
  vcl::backend::mem_handle::cuda_handle_type cuda_handle(cuda_handle_);
  m.cuda_handle() = cuda_handle;
}

void init_cuda_handle(vcl::backend::mem_handle& m,
                      vcl::vcl_size_t ptr, vcl::vcl_size_t raw_size) {
  set_cuda_handle(m, ptr);
  m.raw_size(raw_size);
  m.switch_active_handle_id(vcl::CUDA_MEMORY);
}
#endif

PYVCL_SUBMODULE(platform_support)
{

  bp::enum_<vcl::memory_types>("memory_types")
    ENUM_VALUE(vcl, MEMORY_NOT_INITIALIZED)
    ENUM_VALUE(vcl, MAIN_MEMORY)
    ENUM_VALUE(vcl, OPENCL_MEMORY)
    ENUM_VALUE(vcl, CUDA_MEMORY)
    ;

  bp::scope().attr("default_memory_type") = vcl::backend::default_memory_type();

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::backend::mem_handle,
                                  vcl::vcl_size_t,
                                  raw_size,
                                  mem_handle_get_raw_size,
                                  () const);
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::backend::mem_handle,
                                  void,
                                  raw_size,
                                  mem_handle_set_raw_size,
                                  (vcl::vcl_size_t));
  bp::class_<vcl::backend::mem_handle>("mem_handle")
    .def("init_ram_handle", init_ram_handle)
    .add_property("ram_handle", get_ram_handle, set_ram_handle)
#ifdef VIENNACL_WITH_OPENCL
    .def("init_opencl_handle", init_opencl_handle)
    .add_property("opencl_handle", get_opencl_handle, set_opencl_handle)
    .add_property("opencl_context",
                  bp::make_function(get_opencl_handle_context,
                                    bp::return_value_policy<bp::reference_existing_object>()),
                  set_opencl_handle_context)
#endif
#ifdef VIENNACL_WITH_CUDA
    .def("init_cuda_handle", init_cuda_handle)
    .add_property("cuda_handle", get_cuda_handle, set_cuda_handle)
#endif
    .add_property("active_handle_id", &vcl::backend::mem_handle::get_active_handle_id,
                  &vcl::backend::mem_handle::switch_active_handle_id)
    .add_property("raw_size", mem_handle_get_raw_size, mem_handle_set_raw_size)
    .def(bp::self == vcl::backend::mem_handle())
    .def(bp::self != vcl::backend::mem_handle())
    .def(bp::self < vcl::backend::mem_handle())
    .def("swap", &vcl::backend::mem_handle::swap)
    ;

  bp::class_<vcl::context>("context")
    .def(bp::init<vcl::memory_types>())
#ifdef VIENNACL_WITH_OPENCL
    .def(bp::init<const vcl::ocl::context&>())
    .add_property("opencl_context",
                  bp::make_function(&vcl::context::opencl_context,
                                    bp::return_value_policy<bp::reference_existing_object>()))
#endif
    .add_property("memory_type", &vcl::context::memory_type)
    ;

}
