#ifndef _PYVIENNACL_H
#define _PYVIENNACL_H

#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include <viennacl/tools/shared_ptr.hpp>
#include <viennacl/matrix.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/norm_frobenius.hpp>
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
  op_add,
  op_sub,
  op_mul,
  op_div,
  op_iadd,
  op_isub,
  op_imul,
  op_idiv,
  op_inner_prod,
  op_outer_prod,
  op_element_prod,
  op_element_pow,
  op_element_div,
  op_norm_1,
  op_norm_2,
  op_norm_inf,
  op_norm_frobenius,
  op_index_norm_inf,
  op_plane_rotation,
  op_trans,
  op_prod,
  op_solve,
  op_solve_precond,
  op_inplace_solve
};

// Generic operation dispatch class -- see specialisations below
template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op, int PyObjs>
struct pyvcl_worker
{
  static ReturnT do_op(void* o) {}
};

// This class wraps operations in a type-independent way up to 4 operands.
// It's mainly used to simplify and consolidate calling conventions in the 
// main module code far below, but it also includes a small amount of logic
// for the extraction of C++ types from Python objects where necessary.
//
// Ultimately, I may well do away with this, and interface with the kernel
// scheduler directly. But this is a useful start to have, in order to get
// a working prototype.
//
template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op, int PyObjs=0>
struct pyvcl_op
{
  Operand1T operand1;
  Operand2T operand2;
  Operand3T operand3;
  Operand4T operand4;
  friend struct pyvcl_worker<ReturnT,
			     Operand1T, Operand2T,
			     Operand3T, Operand4T,
			     op, PyObjs>;
  
  pyvcl_op(Operand1T opand1, Operand2T opand2,
	   Operand3T opand3, Operand4T opand4)
    : operand1(opand1), operand2(opand2),
      operand3(opand3), operand4(opand4)
  {
    
    /*
      
      The value of the template variable PyObjs determines which operands
      need to be extracted from Python objects, by coding the operand
      "position" in binary. This is the object-extraction logic alluded to
      in the comments above.
      
      So, given (as an example) PyObjs == 7 == 0111b, and given that we 
      number operands from left to right, the following operands need
      extraction: operand2, operand3, and operand4.
      
    */

    /*    
    if (PyObjs & 8) {
      operand1 = static_cast<Operand1T>
	(bp::extract<Operand1T>((bp::api::object)opand1));
    }
    
    if (PyObjs & 4) {
      operand2 = static_cast<Operand2T>
	(bp::extract<Operand2T>((bp::api::object)opand2));
    }
    
    if (PyObjs & 2) {
      operand3 = static_cast<Operand3T>
	(bp::extract<Operand3T>((bp::api::object)opand3));
    }
    
    if (PyObjs & 1) {
      operand4 = static_cast<Operand4T>
	(bp::extract<Operand4T>((bp::api::object)opand4));
    }
    //*/
  }    

  // Should I just use operator(), I wonder..
  ReturnT do_op()
  {
    return pyvcl_worker<ReturnT,
			Operand1T, Operand2T,
			Operand3T, Operand4T,
			op, PyObjs>::do_op(*this);
  }
};

// Convenient operation dispatch functions.
// These functions make setting up and calling the pyvcl_op class much
// simpler for the specific 1-, 2-, 3- and 4-operand cases.

template <class ReturnT,
	  class Operand1T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_1ary_op(Operand1T a)
{
  pyvcl_op<ReturnT,
	   Operand1T, NoneT,
	   NoneT, NoneT,
	   op, PyObjs>
    o (a, NULL, NULL, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_2ary_op(Operand1T a, Operand2T b)
{
  pyvcl_op<ReturnT,
	   Operand1T, Operand2T,
	   NoneT, NoneT,
	   op, PyObjs>
    o (a, b, NULL, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_3ary_op(Operand1T a, Operand2T b, Operand3T c)
{
  pyvcl_op<ReturnT,
	   Operand1T, Operand2T,
	   Operand3T, NoneT,
	   op, PyObjs>
    o (a, b, c, NULL);
  return o.do_op();
}

template <class ReturnT,
	  class Operand1T, class Operand2T,
	  class Operand3T, class Operand4T,
	  op_t op, int PyObjs>
ReturnT pyvcl_do_4ary_op(Operand1T a, Operand2T b,
			 Operand3T c, Operand4T d)
  {
    pyvcl_op<ReturnT,
	     Operand1T, Operand2T,
	     Operand3T, Operand4T,
	     op, PyObjs>
      o (a, b, c, d);
  return o.do_op();
}


/*****************************
  Operation wrapper functions
 *****************************/

// These macros define specialisations of the pyvcl_worker class
// which is used to dispatch viennacl operations.

#define OP_TEMPLATE template <class ReturnT,                    \
                              class Operand1T, class Operand2T, \
                              class Operand3T, class Operand4T, \
                              int PyObjs>
#define PYVCL_WORKER_STRUCT(OP) OP_TEMPLATE                \
  struct pyvcl_worker<ReturnT,                             \
                      Operand1T, Operand2T,                \
                      Operand3T, Operand4T,                \
                      OP, PyObjs>
#define DO_OP_FUNC(OP) PYVCL_WORKER_STRUCT(OP) {                     \
  static ReturnT do_op(pyvcl_op<ReturnT,                             \
                       Operand1T, Operand2T,                         \
                       Operand3T, Operand4T,                         \
                       OP, PyObjs>& o)

// And the actual operations follow below.

DO_OP_FUNC(op_add) { return o.operand1 + o.operand2; } };
DO_OP_FUNC(op_sub) { return o.operand1 - o.operand2; } };
DO_OP_FUNC(op_mul) { return o.operand1 * o.operand2; } };
DO_OP_FUNC(op_div) { return o.operand1 / o.operand2; } };

DO_OP_FUNC(op_inner_prod)
{
  return vcl::linalg::inner_prod(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_outer_prod)
{
  return vcl::linalg::outer_prod(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_element_prod)
{
  return vcl::linalg::element_prod(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_element_pow)
{
  return vcl::linalg::element_pow(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_element_div)
{
  return vcl::linalg::element_div(o.operand1, o.operand2);
} };

DO_OP_FUNC(op_iadd)
{
  o.operand1 += o.operand2;
  return o.operand1;
} };

DO_OP_FUNC(op_isub)
{
  o.operand1 -= o.operand2;
  return o.operand1;
} };

DO_OP_FUNC(op_imul)
{
  o.operand1 *= o.operand2;
  return o.operand1;
} };

DO_OP_FUNC(op_idiv)
{
  o.operand1 /= o.operand2;
  return o.operand1;
} };

DO_OP_FUNC(op_norm_1)
{
  return vcl::linalg::norm_1(o.operand1);
} };

DO_OP_FUNC(op_norm_2)
{
  return vcl::linalg::norm_2(o.operand1);
} };

DO_OP_FUNC(op_norm_inf)
{
  return vcl::linalg::norm_inf(o.operand1);
} };

DO_OP_FUNC(op_norm_frobenius)
{
  return vcl::linalg::norm_frobenius(o.operand1);
} };

DO_OP_FUNC(op_index_norm_inf)
{
  return vcl::linalg::index_norm_inf(o.operand1);
} };

DO_OP_FUNC(op_plane_rotation)
{
  vcl::linalg::plane_rotation(o.operand1, o.operand2,
			      o.operand3, o.operand4);
  return bp::object();
} };

DO_OP_FUNC(op_trans)
{
  return vcl::trans(o.operand1);
} };

DO_OP_FUNC(op_prod)
{
  return vcl::linalg::prod(o.operand1, o.operand2);
} };

/** @brief Returns a double describing the VCL_T */
template <class HostT>
HostT vcl_scalar_to_host(const vcl::scalar<HostT>& vcl_s)
{
  HostT cpu_s = vcl_s;
  return cpu_s;
}

#define DISAMBIGUATE_FUNCTION_PTR(RET, OLD_NAME, NEW_NAME, ARGS) \
  RET (*NEW_NAME) ARGS = &OLD_NAME;

#define DISAMBIGUATE_CLASS_FUNCTION_PTR(CLASS, RET, OLD_NAME, NEW_NAME, ARGS)\
  RET (CLASS::*NEW_NAME) ARGS = &CLASS::OLD_NAME;


/**************
   Submodules
 **************/

#define PYVCL_SUBMODULE(NAME) void export_ ## NAME ()

PYVCL_SUBMODULE(vector_int);
PYVCL_SUBMODULE(vector_long);
PYVCL_SUBMODULE(vector_uint);
PYVCL_SUBMODULE(vector_ulong);
PYVCL_SUBMODULE(vector_float);
PYVCL_SUBMODULE(vector_double);

PYVCL_SUBMODULE(dense_matrix_int);
PYVCL_SUBMODULE(dense_matrix_long);
PYVCL_SUBMODULE(dense_matrix_uint);
PYVCL_SUBMODULE(dense_matrix_ulong);
PYVCL_SUBMODULE(dense_matrix_float);
PYVCL_SUBMODULE(dense_matrix_double);

PYVCL_SUBMODULE(compressed_matrix);
PYVCL_SUBMODULE(coordinate_matrix);
PYVCL_SUBMODULE(ell_matrix);
PYVCL_SUBMODULE(hyb_matrix);

PYVCL_SUBMODULE(direct_solvers);
PYVCL_SUBMODULE(iterative_solvers);
PYVCL_SUBMODULE(eig);
PYVCL_SUBMODULE(extra_functions);
PYVCL_SUBMODULE(scheduler);

PYVCL_SUBMODULE(opencl_support);


#endif
