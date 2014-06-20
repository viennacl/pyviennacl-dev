#include "common.hpp"

#include <viennacl/context.hpp>
#include <viennacl/backend/mem_handle.hpp>

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
                                  vcl::backend::mem_handle::ram_handle_type&,
                                  ram_handle,
                                  ram_handle,
                                  ());
#ifdef VIENNACL_WITH_OPENCL
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::backend::mem_handle,
                                  vcl::ocl::handle<cl_mem>&,
                                  opencl_handle,
                                  opencl_handle,
                                  ());
#endif
#ifdef VIENNACL_WITH_CUDA
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::backend::mem_handle,
                                  vcl::backend::mem_handle::cuda_handle_type&,
                                  cuda_handle,
                                  cuda_handle,
                                  ());
#endif
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
    .add_property("ram_handle", 
                  bp::make_function(ram_handle, bp::return_internal_reference<>()))
#ifdef VIENNACL_WITH_OPENCL
    .add_property("opencl_handle", 
                  bp::make_function(opencl_handle, bp::return_internal_reference<>()))
#endif
#ifdef VIENNACL_WITH_CUDA
    .add_property("cuda_handle", 
                  bp::make_function(cuda_handle, bp::return_internal_reference<>()))
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
