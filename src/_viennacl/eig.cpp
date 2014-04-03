#include "viennacl.h"
#include "sparse_matrix.h"

#include <viennacl/linalg/power_iter.hpp>
#include <viennacl/linalg/lanczos.hpp>

PYVCL_SUBMODULE(eig)
{

  // Tag class definitions

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::power_iter_tag, double,
                                  factor, get_power_iter_factor, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::power_iter_tag, std::size_t,
                                  max_iterations,
                                  get_power_iter_max_iterations, () const)
  bp::class_<vcl::linalg::power_iter_tag>("power_iter_tag")
    .def(bp::init<double, std::size_t>())
    .add_property("factor", get_power_iter_factor)
    .add_property("max_iterations", get_power_iter_max_iterations)
    ;

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::lanczos_tag, std::size_t,
                                  num_eigenvalues,
                                  get_lanczos_num_eigenvalues, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::lanczos_tag, double,
                                  factor, get_lanczos_factor, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::lanczos_tag, std::size_t,
                                  krylov_size, get_lanczos_krylov_size,
                                  () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::lanczos_tag, int,
                                  method, get_lanczos_method, () const)
  bp::class_<vcl::linalg::lanczos_tag>("lanczos_tag")
    .def(bp::init<double, std::size_t, int, std::size_t>())
    .add_property("num_eigenvalues", get_lanczos_num_eigenvalues)
    .add_property("factor", get_lanczos_factor)
    .add_property("krylov_size", get_lanczos_krylov_size)
    .add_property("method", get_lanczos_method)
    ;


  // Dense matrices

  DISAMBIGUATE_FUNCTION_PTR(double, 
                            vcl::linalg::eig,eig_power_iter_double_row,
                            (const vcl::matrix<double, vcl::row_major>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_double_row);

  DISAMBIGUATE_FUNCTION_PTR(float, 
                            vcl::linalg::eig,eig_power_iter_float_row,
                            (const vcl::matrix<float, vcl::row_major>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_float_row);

  DISAMBIGUATE_FUNCTION_PTR(double, 
                            vcl::linalg::eig,eig_power_iter_double_col,
                            (const vcl::matrix<double, vcl::column_major>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_double_col);

  DISAMBIGUATE_FUNCTION_PTR(float, 
                            vcl::linalg::eig,eig_power_iter_float_col,
                            (const vcl::matrix<float, vcl::column_major>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_float_col);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<double>,
                            vcl::linalg::eig, eig_lanczos_vector_double_row,
                            (const vcl::matrix<double, vcl::row_major>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_double_row);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<float>,
                            vcl::linalg::eig, eig_lanczos_vector_float_row,
                            (const vcl::matrix<float, vcl::row_major>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_float_row);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<double>,
                            vcl::linalg::eig, eig_lanczos_vector_double_col,
                            (const vcl::matrix<double, vcl::column_major>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_double_col);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<float>,
                            vcl::linalg::eig, eig_lanczos_vector_float_col,
                            (const vcl::matrix<float, vcl::column_major>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_float_col);


  // Sparse matrices -- Lanczos

  DISAMBIGUATE_FUNCTION_PTR(std::vector<double>,
                            vcl::linalg::eig, eig_lanczos_vector_double_compressed,
                            (const vcl::compressed_matrix<double>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_double_compressed);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<float>,
                            vcl::linalg::eig, eig_lanczos_vector_float_compressed,
                            (const vcl::compressed_matrix<float>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_float_compressed);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<double>,
                            vcl::linalg::eig, eig_lanczos_vector_double_coordinate,
                            (const vcl::coordinate_matrix<double>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_double_coordinate);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<float>,
                            vcl::linalg::eig, eig_lanczos_vector_float_coordinate,
                            (const vcl::coordinate_matrix<float>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_float_coordinate);

  /*
  DISAMBIGUATE_FUNCTION_PTR(std::vector<double>,
                            vcl::linalg::eig, eig_lanczos_vector_double_ell,
                            (const vcl::ell_matrix<double>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_double_ell);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<float>,
                            vcl::linalg::eig, eig_lanczos_vector_float_ell,
                            (const vcl::ell_matrix<float>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_float_ell);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<double>,
                            vcl::linalg::eig, eig_lanczos_vector_double_hyb,
                            (const vcl::hyb_matrix<double>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_double_hyb);

  DISAMBIGUATE_FUNCTION_PTR(std::vector<float>,
                            vcl::linalg::eig, eig_lanczos_vector_float_hyb,
                            (const vcl::hyb_matrix<float>&, 
                             const vcl::linalg::lanczos_tag&))
  bp::def("eig", eig_lanczos_vector_float_hyb);
  */

  // Sparse matrices -- power_iter

  DISAMBIGUATE_FUNCTION_PTR(double,
                            vcl::linalg::eig, eig_power_iter_double_compressed,
                            (const vcl::compressed_matrix<double>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_double_compressed);

  DISAMBIGUATE_FUNCTION_PTR(float,
                            vcl::linalg::eig, eig_power_iter_float_compressed,
                            (const vcl::compressed_matrix<float>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_float_compressed);

  DISAMBIGUATE_FUNCTION_PTR(double,
                            vcl::linalg::eig, eig_power_iter_double_coordinate,
                            (const vcl::coordinate_matrix<double>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_double_coordinate);

  DISAMBIGUATE_FUNCTION_PTR(float,
                            vcl::linalg::eig, eig_power_iter_float_coordinate,
                            (const vcl::coordinate_matrix<float>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_float_coordinate);

  /*
  DISAMBIGUATE_FUNCTION_PTR(double,
                            vcl::linalg::eig, eig_power_iter_double_ell,
                            (const vcl::ell_matrix<double>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_double_ell);

  DISAMBIGUATE_FUNCTION_PTR(float,
                            vcl::linalg::eig, eig_power_iter_float_ell,
                            (const vcl::ell_matrix<float>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_float_ell);

  DISAMBIGUATE_FUNCTION_PTR(double,
                            vcl::linalg::eig, eig_power_iter_double_hyb,
                            (const vcl::hyb_matrix<double>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_double_hyb);

  DISAMBIGUATE_FUNCTION_PTR(float,
                            vcl::linalg::eig, eig_power_iter_float_hyb,
                            (const vcl::hyb_matrix<float>&, 
                             const vcl::linalg::power_iter_tag&))
  bp::def("eig", eig_power_iter_float_hyb);
  */

}
