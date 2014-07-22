#include "common.hpp"
#include "device_specific.hpp"
#include "scheduler.hpp"

vcl::tools::shared_ptr<ds::statements_container>
make_statements_tuple(bp::list statement_wrappers,
                      ds::statements_container::order_type order) {

  std::list<vcl::scheduler::statement> statements;

  for (long i=0; i < bp::len(statement_wrappers); ++i) {
    const statement_wrapper& w = bp::extract<statement_wrapper>(statement_wrappers[i]);
    statements.push_front(w.get_vcl_statement());
  }

  ds::statements_container* s = new ds::statements_container(statements, order);

  return vcl::tools::shared_ptr<ds::statements_container>(s);
}

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
    bp::scope outer = bp::class_<ds::name, bp::bases<ds::template_base> >(#name, bp::init<ds::name::parameters, ## __VA_ARGS__, std::string const &>()); \
    bp::class_<ds::name::parameters, bp::bases<ds::template_base::parameters> >("parameters", bp::init< parameter_arguments >()); \
  }
  
  WRAP_TEMPLATE(vector_axpy_template,ARGS_LIST(uint,uint,uint,uint));
  WRAP_TEMPLATE(matrix_axpy_template,ARGS_LIST(uint,uint,uint,uint,uint,uint));
  WRAP_TEMPLATE(reduction_template,ARGS_LIST(uint,uint,uint,uint));
  WRAP_TEMPLATE(row_wise_reduction_template,ARGS_LIST(uint,uint,uint,uint), char);
  WRAP_TEMPLATE(matrix_product_template,ARGS_LIST(uint,uint,uint,uint,uint,uint,uint,bool,bool,uint,uint), char, char);

  bp::enum_<ds::statements_container::order_type>
    ("statements_tuple_order_type")
    ENUM_VALUE(ds::statements_container, SEQUENTIAL)
    ENUM_VALUE(ds::statements_container, INDEPENDENT)
    ;

  bp::class_<ds::statements_container,
             vcl::tools::shared_ptr<ds::statements_container> >
    ("statements_tuple", bp::no_init)
    .def("__init__", bp::make_constructor(make_statements_tuple))
    ;

}
                                                                            

