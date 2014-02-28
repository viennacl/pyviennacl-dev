#include "viennacl.h"
#include "vector.h"

#include <vienncl/ocl/context.hpp>
#include <vienncl/ocl/device.hpp>
#include <vienncl/ocl/platform.hpp>


// TODO -- use DISAMBIGUATE_FUNCTION_PTR instead
void setup_context_many(long i, std::vector<vcl::ocl::device> v) {
  vcl::ocl::setup_context(i, v);
}

// TODO -- use DISAMBIGUATE_FUNCTION_PTR instead
void setup_context_single(long i, vcl::ocl::device d) {
  vcl::ocl::setup_context(i, d);
}


PYVCL_SUBMODULE(opencl_support)
{

  bp::class<vcl::ocl::device>("device")
    //.def(bp::init<>())
    .def("name", &vcl::ocl::device::name)
    .def("vendor", &vcl::ocl::device::vendor)
    .def("version", &vcl::ocl::device::version)
    .def("driver_version", &vcl::ocl::device::driver_version)
    .def("info", &vcl::ocl::device::info)
    .def("full_info", &vcl::ocl::device::full_info)
    .def("extensions", &vcl::ocl::device::extensions)
    .def("double_support", &vcl::ocl::device::double_support)
    ;
  
  bp::class<vcl::ocl::platform>("platform", bp::no_init)
    .def("info", &vcl::ocl::platform::info)
    .dev("devices", &vcl::ocl::platform::devices)
    ;

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, void,
                                  platform_index, set_platform_index,
                                  vcl::vcl_size_t);
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, vcl::vcl_size_t,
                                  platform_index, get_platform_index,
                                  void);
  bp::class<vcl::ocl::context>("context")
    .def("init", &vcl::ocl::context::init)
    .def("current_device", &vcl::ocl::context::current_device)
    .def("devices", &vcl::ocl::context::devices)
    .def("add_device", &vcl::ocl::context::add_device)
    .def("switch_device", &vcl::ocl::context::switch_device)
    .def("get_platform_index", get_platform_index)
    .def("set_platform_index", set_platform_index)
    ;

  bp::def("current_context", vcl::ocl::current_context);
  bp::def("current_device", vcl::ocl::current_device);

  bp::def("setup_context", setup_context_single);
  bp::def("setup_context", setup_context_many);

  bp::def("switch_context", vcl::ocl::switch_context);


  /*

    Notes on platform support; esp. OpenCL
    ======================================

    PyOpenCL Integration
    --------------------

    * Want to take advantage of PyOpenCL's int_ptr to access ViennaCL
      objects in other (Py)OpenCL code and vice versa.

    * need to get the underlying OpenCL pointers out of ViennaCL --
      and be able to pass them back in


    Multi-platform support
    ----------------------

    * useful to specify the backend details on creation of a ViennaCL
      object (eg matrix)
      + what about copying between back-ends?

    * how to define 'back-end'? := context?

    * throw an exception if attempting to do an operation across
      back-ends


    Scheduler integration and linking
    ---------------------------------

    * users able to define a custom Node class
      + useful for prototyping algorithm implementations!
      + so: useful to expose an API similar to the underlying
            ViennaCL structure

    * class info determines dtypes, arguments and return type
    
    * class provides kernel (source, binary, whatever supported by
      back-end)
      + device-specific kernel support?

    * PyViennaCL registers custom Node with ViennaCL scheduler,
      assigning the Node an ID

    * ViennaCL then has a mapping of (ID, custom operation) pairs

    * when the scheduler is called with the relevant ID, argument types
      are checked and operation is scheduled for dispatch
      + how is the return type created?

    * posible to harness the generator usefully?
      

   */

}
