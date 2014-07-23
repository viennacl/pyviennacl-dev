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
    #define __PROP(name) .def_readonly(#name, &ds::template_base::parameters_type::name)
    bp::scope outer = bp::class_<ds::template_base, boost::noncopyable>("template_base", bp::no_init);
    bp::class_<ds::template_base::parameters_type>("parameters_type", bp::no_init)
              __PROP(simd_width)
              __PROP(local_size_0)
              __PROP(local_size_1);
    #undef __PROP
  }
  
  #define WRAP_TEMPLATE(name, ...) bp::scope outer = bp::class_<ds::name, bp::bases<ds::template_base> >(#name, bp::init<ds::name::parameters_type, ## __VA_ARGS__, std::string const &>())
  #define WRAP_PARAMETERS(name, ...) bp::class_<ds::name::parameters_type, bp::bases<ds::template_base::parameters_type> >("parameters_type", bp::init< __VA_ARGS__ >())
  #define __PROP_BASE(name, tpname) .def_readonly(#name, &ds::tpname::parameters_type::name)
  //Vector AXPY
  { 
    #define __PROP(name) __PROP_BASE(name, vector_axpy_template)
    WRAP_TEMPLATE(vector_axpy_template);
    WRAP_PARAMETERS(vector_axpy_template, uint, size_t, size_t, uint)
        __PROP(num_groups) __PROP(decomposition);
    #undef __PROP
  }
  
  //Matrix AXPY
  { 
    #define __PROP(name) __PROP_BASE(name, matrix_axpy_template)
    WRAP_TEMPLATE(matrix_axpy_template);
    WRAP_PARAMETERS(matrix_axpy_template, uint, size_t, size_t, size_t, size_t, uint)
        __PROP(num_groups_0) __PROP(num_groups_1)  __PROP(decomposition);
    #undef __PROP
  }
  
  //Reduction
  { 
    #define __PROP(name) __PROP_BASE(name, reduction_template)
    WRAP_TEMPLATE(reduction_template);
    WRAP_PARAMETERS(reduction_template, uint, size_t, size_t, uint)
        __PROP(num_groups)  __PROP(decomposition);
    #undef __PROP
  }
  
  //Row-wise reduction
  { 
    #define __PROP(name) __PROP_BASE(name, row_wise_reduction_template)
    WRAP_TEMPLATE(row_wise_reduction_template, char);
    WRAP_PARAMETERS(row_wise_reduction_template, uint, size_t, size_t, uint)
        __PROP(num_groups_0);
    #undef __PROP
  }
  
  //Matrix product
  { 
    #define __PROP(name) __PROP_BASE(name, matrix_product_template)
    WRAP_TEMPLATE(matrix_product_template, char, char);
    WRAP_PARAMETERS(matrix_product_template, uint, size_t, uint, size_t, uint, uint, uint, bool, bool, size_t, size_t)
        __PROP(kL) __PROP(mS) __PROP(kS) __PROP(nS)
        __PROP(use_A_local) __PROP(use_B_local)
        __PROP(local_fetch_0) __PROP(local_fetch_1)
        __PROP(mL) __PROP(nL);
    #undef __PROP
  }
 
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
                                                                            

