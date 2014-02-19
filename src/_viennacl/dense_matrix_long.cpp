#include "dense_matrix.h"

PYVCL_SUBMODULE(dense_matrix_long)
{
  EXPORT_DENSE_MATRIX_CLASS(long, row, vcl::row_major, ublas::row_major)
  EXPORT_DENSE_MATRIX_CLASS(long, col, vcl::column_major, ublas::column_major)
}

