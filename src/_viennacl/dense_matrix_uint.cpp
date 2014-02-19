#include "dense_matrix.h"

PYVCL_SUBMODULE(dense_matrix_uint)
{
  EXPORT_DENSE_MATRIX_CLASS(uint, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(uint, col, vcl::column_major, ublas::column_major)
}

