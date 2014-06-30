#include "compressed_compressed_matrix.hpp"

PYVCL_SUBMODULE(compressed_compressed_matrix)
{
  EXPORT_COMPRESSED_COMPRESSED_MATRIX(float);
  EXPORT_COMPRESSED_COMPRESSED_MATRIX(double);
  // TODO: other types
}

