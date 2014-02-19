#ifndef _PYVIENNACL_ENTRY_PROXY_H
#define _PYVIENNACL_ENTRY_PROXY_H

#include "viennacl.h"
#include <viennacl/forwards.h>

// THIS IS NOT VERY C++ !!

template <class SCALARTYPE, class VCLTYPE>
SCALARTYPE get_vcl_vector_entry(VCLTYPE o, vcl::vcl_size_t x)
{
  return o(x);
}

template <class SCALARTYPE, class VCLTYPE>
SCALARTYPE get_vcl_matrix_entry(VCLTYPE o, vcl::vcl_size_t x, vcl::vcl_size_t y)
{
  return o(x, y);
}

template <class SCALARTYPE, class VCLTYPE>
bp::object set_vcl_vector_entry(VCLTYPE o, vcl::vcl_size_t x, SCALARTYPE v)
{
  o(x) = v;

  return bp::object();
}

template <class SCALARTYPE, class VCLTYPE>
bp::object set_vcl_matrix_entry(VCLTYPE o, vcl::vcl_size_t x, vcl::vcl_size_t y, SCALARTYPE v)
{
  o(x, y) = v;

  return bp::object();
}

#endif 
