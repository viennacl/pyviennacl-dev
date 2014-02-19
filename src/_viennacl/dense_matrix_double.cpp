#include "dense_matrix.h"

PYVCL_SUBMODULE(dense_matrix_double)
{
  EXPORT_DENSE_MATRIX_CLASS(double, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(double, col, vcl::column_major, ublas::column_major)
}

