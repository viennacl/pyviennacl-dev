#ifndef _PYVIENNACL_H
#define _PYVIENNACL_H

#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include <viennacl/tools/shared_ptr.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/prod.hpp>

#define CONCAT(...) __VA_ARGS__

namespace vcl = viennacl;
namespace bp = boost::python;
namespace np = boost::numpy;

typedef void* NoneT;

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

namespace viennacl {
  namespace tools {
    template <typename T> T* get_pointer(vcl::tools::shared_ptr<T> const& p) {
      return p.get();
    }
  }
}

namespace boost {
  namespace python {
    template <typename T> struct pointee<vcl::tools::shared_ptr<T> > {
      typedef T type;
    };
  }
}

// TODO: Use ViennaCL operation tags?
enum op_t {
  op_inner_prod=0,   //  0
  op_outer_prod,     //  1
  op_element_pow,    //  2
  op_norm_1,         //  3
  op_norm_2,         //  4
  op_norm_inf,       //  5
  op_norm_frobenius, //  6
  op_index_norm_inf, //  7
  op_plane_rotation, //  8
  op_prod,           //  9
  op_solve,          // 10
  op_solve_precond,  // 11
  op_inplace_solve,  // 12
  op_fft_2d,         // 13
  op_ifft_2d,        // 14
  op_inplace_fft_2d, // 15
  op_inplace_ifft_2d,// 16
  op_convolve_2d,    // 17
  op_convolve_i_2d,  // 18
  op_fft_3d,         // 19
  op_inplace_fft_3d, // 20
  op_fft_normalize_2d, // 21
  op_inplace_qr,     // 22
  op_inplace_qr_apply_trans_q, // 23
  op_recoverq,       // 24
  op_nmf,            // 25
  op_svd,            // 26
  op_diag,           // 27
  op_row,            // 28
  op_column          // 29
};

// Generic operation dispatch class -- specialised for different ops
template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op>
struct pyvcl_worker
{
  static ReturnT do_op(void* o) {}
};

// This class wraps operations in a type-independent way up to 4 operands.
// It's mainly used to simplify and consolidate calling conventions in the 
// main module code.
template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op>
struct pyvcl_op
{
  Operand1T operand1;
  Operand2T operand2;
  Operand3T operand3;
  Operand4T operand4;
  friend struct pyvcl_worker<ReturnT,
			     Operand1T, Operand2T,
			     Operand3T, Operand4T,
			     op>;
  
  pyvcl_op(Operand1T opand1, Operand2T opand2,
	   Operand3T opand3, Operand4T opand4)
    : operand1(opand1), operand2(opand2),
      operand3(opand3), operand4(opand4)
  { }

  // Should I just use operator(), I wonder..
  ReturnT do_op()
  {
    return pyvcl_worker<ReturnT,
			Operand1T, Operand2T,
			Operand3T, Operand4T,
			op>::do_op(*this);
  }
};

// Convenient operation dispatch functions.
// These functions make setting up and calling the pyvcl_op class much
// simpler for the specific 1-, 2-, 3- and 4-operand cases.

template <class ReturnT,
	  class Operand1T,
	  op_t op>
ReturnT pyvcl_do_1ary_op(Operand1T a)
{
  pyvcl_op<ReturnT,
	   Operand1T, NoneT,
	   NoneT, NoneT,
           op>
    o (a, NULL, NULL, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  op_t op>
ReturnT pyvcl_do_2ary_op(Operand1T a, Operand2T b)
{
  pyvcl_op<ReturnT,
	   Operand1T, Operand2T,
	   NoneT, NoneT,
	   op>
    o (a, b, NULL, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T,
	  op_t op>
ReturnT pyvcl_do_3ary_op(Operand1T a, Operand2T b, Operand3T c)
{
  pyvcl_op<ReturnT,
	   Operand1T, Operand2T,
	   Operand3T, NoneT,
	   op>
    o (a, b, c, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op>
ReturnT pyvcl_do_4ary_op(Operand1T a, Operand2T b,
			 Operand3T c, Operand4T d)
  {
    pyvcl_op<ReturnT,
	     Operand1T, Operand2T,
	     Operand3T, Operand4T,
	     op>
      o (a, b, c, d);
  return o.do_op();
}

// These macros define specialisations of the pyvcl_worker class
// which is used to dispatch viennacl operations.

#define OP_TEMPLATE template <class ReturnT,                    \
                              class Operand1T, class Operand2T, \
                              class Operand3T, class Operand4T>
#define PYVCL_WORKER_STRUCT(OP) OP_TEMPLATE                \
  struct pyvcl_worker<ReturnT,                             \
                      Operand1T, Operand2T,                \
                      Operand3T, Operand4T,                \
                      OP>
#define DO_OP_FUNC(OP) PYVCL_WORKER_STRUCT(OP) {                     \
  static ReturnT do_op(pyvcl_op<ReturnT,                             \
                       Operand1T, Operand2T,                         \
                       Operand3T, Operand4T,                         \
                       OP>& o)                                       \

#define CLOSE_OP_FUNC }

DO_OP_FUNC(op_prod)
{
  return vcl::linalg::prod(o.operand1, o.operand2);
}
CLOSE_OP_FUNC;

#define COMMA ,

#define DISAMBIGUATE_FUNCTION_PTR(RET, OLD_NAME, NEW_NAME, ARGS) \
  RET (*NEW_NAME) ARGS = &OLD_NAME;

#define DISAMBIGUATE_CLASS_FUNCTION_PTR(CLASS, RET, OLD_NAME, NEW_NAME, ARGS)\
  RET (CLASS::*NEW_NAME) ARGS = &CLASS::OLD_NAME;

#define ENUM_VALUE(NS, V) .value( #V, NS :: V )

#define PYVCL_SUBMODULE(NAME) void export_##NAME()

#define PYTHON_SCOPE_SUBMODULE(NAME)                                    \
  bp::object NAME##_submodule                                           \
  (bp::handle<>(bp::borrowed(PyImport_AddModule("_viennacl." #NAME)))); \
  bp::scope().attr(#NAME) = NAME##_submodule;                           \
  bp::scope NAME##_scope = NAME##_submodule;

#endif
