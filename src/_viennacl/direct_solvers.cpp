#include "direct_solvers.hpp"

PYVCL_SUBMODULE(direct_solvers)
{
  bp::class_<vcl::linalg::lower_tag>("lower_tag");
  bp::class_<vcl::linalg::unit_lower_tag>("unit_lower_tag");
  bp::class_<vcl::linalg::upper_tag>("upper_tag");
  bp::class_<vcl::linalg::unit_upper_tag>("unit_upper_tag");

  EXPORT_DENSE_DIRECT_SOLVERS(float);
  EXPORT_DENSE_DIRECT_SOLVERS(double);
  EXPORT_DENSE_DIRECT_SOLVERS(int);
  EXPORT_DENSE_DIRECT_SOLVERS(uint);
  EXPORT_DENSE_DIRECT_SOLVERS(long);
  EXPORT_DENSE_DIRECT_SOLVERS(ulong);

  EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(vcl::compressed_matrix<float>, float);
  EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(vcl::compressed_matrix<double>, double);

  //EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(vcl::compressed_compressed_matrix<float>, float);
  //EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(vcl::compressed_compressed_matrix<double>, double);

  //EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(vcl::coordinate_matrix<float>, float);
  //EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(vcl::coordinate_matrix<double>, double);

  //EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(vcl::ell_matrix<float>, float);
  //EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(vcl::ell_matrix<double>, double);

  //EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(vcl::sliced_ell_matrix<float>, float);
  //EXPORT_MATRIX_VECTOR_DIRECT_INPLACE_SOLVERS(vcl::sliced_ell_matrix<double>, double);

}

