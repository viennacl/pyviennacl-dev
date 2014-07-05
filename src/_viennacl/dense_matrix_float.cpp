#include "dense_matrix.hpp"

PYVCL_SUBMODULE(dense_matrix_float)
{
  //EXPORT_DENSE_MATRIX_BASE_CLASS(float)
  EXPORT_DENSE_MATRIX_CLASS(float, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(float, col, vcl::column_major, ublas::column_major)
}

