#include "viennacl.h"

#include <viennacl/context.hpp>

PYVCL_SUBMODULE(platform_support)
{

  bp::enum_<vcl::memory_types>("memory_types")
    ENUM_VALUE(vcl, MEMORY_NOT_INITIALIZED)
    ENUM_VALUE(vcl, MAIN_MEMORY)
    ENUM_VALUE(vcl, OPENCL_MEMORY)
    ENUM_VALUE(vcl, CUDA_MEMORY)
    ;

  bp::class_<vcl::context>("context")
    .def(bp::init<vcl::memory_types>())
    //#ifdef VIENNACL_WITH_OPENCL
    .def(bp::init<const vcl::ocl::context&>())
    //#endif
    .def("memory_type", &vcl::context::memory_type)
    ;

}
