#include "common.hpp"
#include "device_specific.hpp"

PYVCL_SUBMODULE(device_specific)
{
  
  //Base
  bp::class_<ds::template_base>("template_base", bp::no_init)
      .def("is_invalid",&ds::template_base::is_invalid);

  bp::class_<ds::vector_axpy_template::parameters
            ,bp::bases<ds::template_base::parameters> >("vector_axpy_template_parameters", bp::init<const char *, uint, uint, uint, uint>());

  bp::class_<ds::matrix_axpy_template::parameters
          ,bp::bases<ds::template_base::parameters> >("matrix_axpy_template_parameters", bp::init<const char *, uint, uint, uint, uint, uint, uint>());
    
  bp::class_<ds::reduction_template::parameters
          ,bp::bases<ds::template_base::parameters> >("reduction_template_parameters", bp::init<const char *, uint, uint, uint, uint>());
    
  bp::class_<ds::row_wise_reduction_template::parameters
        ,bp::bases<ds::template_base::parameters> >("row_wise_reduction_template_parameters", bp::init<const char *, char, uint, uint, uint, uint>());
        
  bp::class_<ds::matrix_product_template::parameters
        ,bp::bases<ds::template_base::parameters> >("matrix_product_template_parameters", bp::init<const char *, char, char,
                                                                            uint, uint, uint, uint, uint, uint, uint,
                                                                            bool, bool, uint, uint>());
            
                                                                            
}
                                                                            

