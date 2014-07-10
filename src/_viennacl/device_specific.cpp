#include "common.hpp"
#include "device_specific.hpp"

PYVCL_SUBMODULE(device_specific)
{
  
  //Base
  {
    bp::scope outer = bp::class_<ds::template_base, boost::noncopyable>("template_base", bp::no_init);
    bp::class_<ds::template_base::parameters>("parameters", bp::no_init);
  }
  
  #define ARGS_LIST(...) __VA_ARGS__
  #define WRAP_TEMPLATE(name, parameter_arguments, ...) \
  { \
    bp::scope outer = bp::class_<ds::name, bp::bases<ds::template_base> >(#name, bp::init<ds::name::parameters, ## __VA_ARGS__, std::string const &>());\
    bp::class_<ds::name::parameters, bp::bases<ds::template_base::parameters> >("parameters", bp::init< parameter_arguments >());\
  }
  
  WRAP_TEMPLATE(vector_axpy_template,ARGS_LIST(uint,uint,uint,uint))
  WRAP_TEMPLATE(matrix_axpy_template,ARGS_LIST(uint,uint,uint,uint,uint,uint))
  WRAP_TEMPLATE(reduction_template,ARGS_LIST(uint,uint,uint,uint))
  WRAP_TEMPLATE(row_wise_reduction_template,ARGS_LIST(uint,uint,uint,uint), char)
  WRAP_TEMPLATE(matrix_product_template,ARGS_LIST(uint,uint,uint,uint,uint,uint,uint,bool,bool,uint,uint), char, char)                                                               
}
                                                                            

