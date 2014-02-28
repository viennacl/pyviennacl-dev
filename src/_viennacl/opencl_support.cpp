#include "viennacl.h"
#include "vector.h"

#include <viennacl/ocl/context.hpp>
#include <viennacl/ocl/device.hpp>
#include <viennacl/ocl/platform.hpp>

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


PYVCL_SUBMODULE(opencl_support)
{

  bp::scope opencl_support_scope = bp::class_<bp::object>("opencl_support", bp::no_init);

  bp::class_<vcl::ocl::platform>("platform", bp::no_init)
    .def("info", &vcl::ocl::platform::info)
    .def("devices", &vcl::ocl::platform::devices)
  ;

  bp::class_<vcl::ocl::device>("device")
    .def("name", &vcl::ocl::device::name)
    .def("vendor", &vcl::ocl::device::vendor)
    .def("version", &vcl::ocl::device::version)
    .def("driver_version", &vcl::ocl::device::driver_version)
    .def("info", &vcl::ocl::device::info)
    .def("full_info", &vcl::ocl::device::full_info)
    .def("extensions", &vcl::ocl::device::extensions)
    .def("double_support", &vcl::ocl::device::double_support)
  ;
  
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, void,
                                  init, init_new_context,
                                  ());
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, void,
                                  add_device, context_add_device,
                                  (viennacl::ocl::device const&));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, void,
                                  switch_device, context_switch_device,
                                  (viennacl::ocl::device const&));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, void,
                                  platform_index, context_set_platform_index,
                                  (vcl::vcl_size_t));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::ocl::context, vcl::vcl_size_t,
                                  platform_index, context_get_platform_index,
                                  () const);
  bp::class_<vcl::ocl::context>("context")
  .def("init", init_new_context)
  .def("current_device", &vcl::ocl::context::current_device, 
       bp::return_value_policy<bp::copy_const_reference>())
  .def("devices", &vcl::ocl::context::devices,
       bp::return_value_policy<bp::copy_const_reference>())
  .def("add_device", context_add_device)
  .def("switch_device", context_switch_device)
  .def("get_platform_index", context_get_platform_index)
  .def("set_platform_index", context_set_platform_index)
  ;
    
  bp::def("current_context", vcl::ocl::current_context,
          bp::return_value_policy<bp::copy_non_const_reference>());
  bp::def("current_device", vcl::ocl::current_device,
          bp::return_value_policy<bp::copy_const_reference>());
    
  DISAMBIGUATE_FUNCTION_PTR(void,
                            vcl::ocl::setup_context,
                            setup_context_single,
                            (long, vcl::ocl::device const&));
  bp::def("setup_context", setup_context_single);
    
  bp::def("switch_context", vcl::ocl::switch_context);

}
