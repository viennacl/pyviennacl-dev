#include "common.hpp"

#include <viennacl/context.hpp>
#include <viennacl/backend/mem_handle.hpp>

vcl::vcl_size_t get_ram_handle(vcl::backend::mem_handle& m) {
  return (vcl::vcl_size_t) m.ram_handle().get();
}
#ifdef VIENNACL_WITH_OPENCL
vcl::vcl_size_t get_opencl_handle(vcl::backend::mem_handle& m) {
  return (vcl::vcl_size_t) m.opencl_handle().get();
}
#endif
#ifdef VIENNACL_WITH_CUDA
vcl::vcl_size_t get_cuda_handle(vcl::backend::mem_handle& m) {
  return (vcl::vcl_size_t) m.cuda_handle().get();
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
    .add_property("ram_handle", get_ram_handle)
#ifdef VIENNACL_WITH_OPENCL
    .add_property("opencl_handle", get_opencl_handle)
#endif
#ifdef VIENNACL_WITH_CUDA
    .add_property("cuda_handle", get_cuda_handle)
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
    //#ifdef VIENNACL_WITH_OPENCL
    .def(bp::init<const vcl::ocl::context&>())
    //#endif
    .def("memory_type", &vcl::context::memory_type)
    ;

}
