#include "structured_matrix.hpp"

PYVCL_SUBMODULE(structured_matrices)
{

  using viennacl::circulant_matrix;
  EXPORT_STRUCTURED_MATRIX(circulant_matrix, float);
  EXPORT_STRUCTURED_MATRIX(circulant_matrix, double);

  using viennacl::hankel_matrix;
  EXPORT_STRUCTURED_MATRIX(hankel_matrix, float);
  EXPORT_STRUCTURED_MATRIX(hankel_matrix, double);

  using viennacl::toeplitz_matrix;
  EXPORT_STRUCTURED_MATRIX(toeplitz_matrix, float);
  EXPORT_STRUCTURED_MATRIX(toeplitz_matrix, double);

  /*
  using viennacl::vandermonde_matrix;
  EXPORT_STRUCTURED_MATRIX(vandermonde_matrix, float);
  EXPORT_STRUCTURED_MATRIX(vandermonde_matrix, double);
  */

}
