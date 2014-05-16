#include <iostream>
#include <typeinfo>

#include "viennacl.h"

#include <viennacl/linalg/cg.hpp>
#include <viennacl/linalg/bicgstab.hpp>
#include <viennacl/linalg/direct_solve.hpp>
#include <viennacl/linalg/gmres.hpp>

/*******************************
  Python module initialisation
 *******************************/
void translate_string_exception(const char* e)
{
  // Use the Python 'C' API to set up an exception object
  PyErr_SetString(PyExc_RuntimeError, e);
}

BOOST_PYTHON_MODULE(_viennacl)
{
  // specify that this module is actually a package
  bp::object package = bp::scope();
  package.attr("__path__") = "_viennacl";

  bp::register_exception_translator<const char*>            
    (&translate_string_exception);                            

  np::initialize();

  // TODO: integrate version into build process
  bp::scope().attr("__version__") = bp::object("1.5.2");

  bp::def("backend_finish", vcl::backend::finish);

#define EXPORT_SCALAR_CLASS(TYPE)                                       \
  bp::class_<vcl::scalar<TYPE> >("scalar_" #TYPE)                       \
    .def(bp::init<TYPE>())                                              \
    .def(bp::init<vcl::scalar<TYPE> >())                                \
    .def("to_host", &vcl_scalar_to_host<TYPE>)

  EXPORT_SCALAR_CLASS(int);
  EXPORT_SCALAR_CLASS(uint);
  EXPORT_SCALAR_CLASS(long);
  EXPORT_SCALAR_CLASS(ulong);
  EXPORT_SCALAR_CLASS(float);
  EXPORT_SCALAR_CLASS(double);

  bp::class_<vcl::range>("range",
                         bp::init<vcl::vcl_size_t, vcl::vcl_size_t>());
  bp::class_<vcl::slice>("slice",
                         bp::init<vcl::vcl_size_t, vcl::vcl_size_t, vcl::vcl_size_t>());

  bp::class_<vcl::linalg::lower_tag>("lower_tag");
  bp::class_<vcl::linalg::unit_lower_tag>("unit_lower_tag");
  bp::class_<vcl::linalg::upper_tag>("upper_tag");
  bp::class_<vcl::linalg::unit_upper_tag>("unit_upper_tag");

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::cg_tag, unsigned int,
                                  iters, get_cg_iters, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::cg_tag, double,
                                  error, get_cg_error, () const)
  bp::class_<vcl::linalg::cg_tag>("cg_tag")
    .def(bp::init<double, vcl::vcl_size_t>())
    .add_property("tolerance", &vcl::linalg::cg_tag::tolerance)
    .add_property("max_iterations", &vcl::linalg::cg_tag::max_iterations)
    .add_property("iters", get_cg_iters)
    .add_property("error", get_cg_error)
    ;

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::bicgstab_tag, vcl::vcl_size_t,
                                  iters, get_bicgstab_iters, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::bicgstab_tag, double,
                                  error, get_bicgstab_error, () const)
  bp::class_<vcl::linalg::bicgstab_tag>("bicgstab_tag")
    .def(bp::init<double, vcl::vcl_size_t, vcl::vcl_size_t>())
    .add_property("tolerance", &vcl::linalg::bicgstab_tag::tolerance)
    .add_property("max_iterations",
                  &vcl::linalg::bicgstab_tag::max_iterations)
    .add_property("max_iterations_before_restart",
                  &vcl::linalg::bicgstab_tag::max_iterations_before_restart)
    .add_property("iters", get_bicgstab_iters)
    .add_property("error", get_bicgstab_error)
    ;

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::gmres_tag, unsigned int,
                                  iters, get_gmres_iters, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::gmres_tag, double,
                                  error, get_gmres_error, () const)
  bp::class_<vcl::linalg::gmres_tag>("gmres_tag")
    .def(bp::init<double, vcl::vcl_size_t, vcl::vcl_size_t>())
    .add_property("tolerance", &vcl::linalg::gmres_tag::tolerance)
    .add_property("max_iterations", &vcl::linalg::gmres_tag::max_iterations)
    .add_property("iters", get_gmres_iters)
    .add_property("error", get_gmres_error)
    .add_property("krylov_dim", &vcl::linalg::gmres_tag::krylov_dim)
    .add_property("max_restarts", &vcl::linalg::gmres_tag::max_restarts)
    ;

  bp::class_<vcl::linalg::no_precond>("no_precond");

export_vector_int();
export_vector_long();
export_vector_uint();
export_vector_ulong();
export_vector_float();
export_vector_double();

export_dense_matrix_int();
export_dense_matrix_long();
export_dense_matrix_uint();
export_dense_matrix_ulong();
export_dense_matrix_float();
export_dense_matrix_double();

export_compressed_matrix();
export_coordinate_matrix();
export_ell_matrix();
export_hyb_matrix();

export_direct_solvers();
export_iterative_solvers();
export_eig();
export_extra_functions();
export_scheduler();

export_opencl_support();

  
}

