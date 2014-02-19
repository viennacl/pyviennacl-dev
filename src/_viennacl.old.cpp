#include <stdint.h>
#include <iostream>
#include <typeinfo>

//#define VIENNACL_DEBUG_BUILD
//#define VIENNACL_DEBUG_ALL
#define VIENNACL_WITH_UBLAS
#define VIENNACL_WITH_OPENCL

#include "_viennacl.h"

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

  bp::register_exception_translator
    <const char*>
    (&translate_string_exception);

  np::initialize();

  // TODO: integrate version into build process
  bp::scope().attr("__version__") = bp::object("1.5.0");

  // --------------------------------------------------

  // *** Utility functions ***
  bp::def("backend_finish", vcl::backend::finish);

  // --------------------------------------------------

  // *** Scalar type ***

  // TODO: EXPOSE ALL NUMERIC TYPES

  bp::class_<vcl::scalar<float> >("scalar_float") // TODO
    // Utility functions
    .def(bp::init<float>())
    .def(bp::init<int>())
    .def("to_host", &vcl_scalar_to_host<float>)
    ;

  bp::class_<vcl::scalar<double> >("scalar_double")
    // Utility functions
    .def(bp::init<double>())
    .def(bp::init<int>())
    .def("to_host", &vcl_scalar_to_host<double>)
    ;

  // --------------------------------------------------

  bp::class_<vcl::range>("range",
                         bp::init<std::size_t, std::size_t>());
  bp::class_<vcl::slice>("slice",
                         bp::init<std::size_t, std::size_t, std::size_t>());

  // *** Vector types ***

  //EXPORT_VECTOR_CLASS(char)
  //EXPORT_VECTOR_CLASS(short)

  export_vector_int();
  export_vector_long();
  export_vector_uint();
  export_vector_ulong();
  export_vector_float();
  export_vector_double();


  // --------------------------------------------------
  
  bp::class_<vcl::linalg::lower_tag>("lower_tag");
  bp::class_<vcl::linalg::unit_lower_tag>("unit_lower_tag");
  bp::class_<vcl::linalg::upper_tag>("upper_tag");
  bp::class_<vcl::linalg::unit_upper_tag>("unit_upper_tag");

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::cg_tag, unsigned int,
                                  iters, get_cg_iters, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::cg_tag, double,
                                  error, get_cg_error, () const)
  bp::class_<vcl::linalg::cg_tag>("cg_tag")
    .def(bp::init<double, unsigned int>())
    .add_property("tolerance", &vcl::linalg::cg_tag::tolerance)
    .add_property("max_iterations", &vcl::linalg::cg_tag::max_iterations)
    .add_property("iters", get_cg_iters)
    .add_property("error", get_cg_error)
    ;

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::bicgstab_tag, std::size_t,
                                  iters, get_bicgstab_iters, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::bicgstab_tag, double,
                                  error, get_bicgstab_error, () const)
  bp::class_<vcl::linalg::bicgstab_tag>("bicgstab_tag")
    .def(bp::init<double, std::size_t, std::size_t>())
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
    .def(bp::init<double, unsigned int, unsigned int>())
    .add_property("tolerance", &vcl::linalg::gmres_tag::tolerance)
    .add_property("max_iterations", &vcl::linalg::gmres_tag::max_iterations)
    .add_property("iters", get_gmres_iters)
    .add_property("error", get_gmres_error)
    .add_property("krylov_dim", &vcl::linalg::gmres_tag::krylov_dim)
    .add_property("max_restarts", &vcl::linalg::gmres_tag::max_restarts)
    ;

  // *** Dense matrix type ***

  /* TODO:::::::::::::::
  EXPORT_DENSE_MATRIX_CLASS(char, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(char, col, vcl::column_major, ublas::column_major)
  EXPORT_DENSE_MATRIX_CLASS(short, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(short, col, vcl::column_major, ublas::column_major)
  EXPORT_DENSE_MATRIX_CLASS(uchar, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(uchar, col, vcl::column_major, ublas::column_major)
  EXPORT_DENSE_MATRIX_CLASS(ushort, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(ushort, col, vcl::column_major, ublas::column_major)
  */

  export_dense_matrix_int();
  export_dense_matrix_long();
  export_dense_matrix_uint();
  export_dense_matrix_ulong();
  export_dense_matrix_float();
  export_dense_matrix_double();

           
  // --------------------------------------------------

  // *** Sparse matrix types ***

  // TODO: Other types than double!
  export_compressed_matrix();
  export_coordinate_matrix();
  export_ell_matrix();
  export_hyb_matrix();

  bp::class_<cpu_compressed_matrix_wrapper<double> >
    ("cpu_compressed_matrix_double")
    .def(bp::init<>())
    .def(bp::init<uint32_t, uint32_t>())
    .def(bp::init<uint32_t, uint32_t, uint32_t>())
    .def(bp::init<cpu_compressed_matrix_wrapper<double> >())
    .def(bp::init<vcl::compressed_matrix<double> >())
    //.def(bp::init<vcl::coordinate_matrix<double> >())
    .def(bp::init<vcl::ell_matrix<double> >())
    .def(bp::init<vcl::hyb_matrix<double> >())
    .def(bp::init<np::ndarray>())
    .def_readonly("nonzeros", &cpu_compressed_matrix_wrapper<double>::places)
    .add_property("nnz", &cpu_compressed_matrix_wrapper<double>::nnz)
    .add_property("size1", &cpu_compressed_matrix_wrapper<double>::size1)
    .add_property("size2", &cpu_compressed_matrix_wrapper<double>::size2)
    .def("resize", &cpu_compressed_matrix_wrapper<double>::resize)
    .def("set", &cpu_compressed_matrix_wrapper<double>::set)
    .def("get", &cpu_compressed_matrix_wrapper<double>::get)
    .def("as_ndarray", &cpu_compressed_matrix_wrapper<double>::as_ndarray)
    .def("as_compressed_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix_with_size<vcl::compressed_matrix<double> >)
    .def("as_coordinate_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix_with_size<vcl::coordinate_matrix<double> >)
    .def("as_ell_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix<vcl::ell_matrix<double> >)
    .def("as_hyb_matrix",
         &cpu_compressed_matrix_wrapper<double>
         ::as_vcl_sparse_matrix<vcl::hyb_matrix<double> >)
    ;

  // --------------------------------------------------

  // Eigenvalue computations
  export_eig();

  // --------------------------------------------------
  
  // More functions (eg, if not supported across all dtypes)
  export_extra_functions();

  // --------------------------------------------------

  // Scheduler interface
  export_scheduler();
    
  // --------------------------------------------------

  // OpenCL interface

  /*

   + vcl::ocl::kernel
   + vcl::ocl::program
   + vcl::ocl::context

   PyOpenCL integration

   */

}

