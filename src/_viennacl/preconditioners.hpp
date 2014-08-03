#ifndef _PYVIENNACL_PRECONDITIONERS_HPP
#define _PYVIENNACL_PRECONDITIONERS_HPP

#include "sparse_matrix.hpp"

#include <viennacl/linalg/ichol.hpp>
#include <viennacl/linalg/ilu.hpp>
#include <viennacl/linalg/jacobi_precond.hpp>
#include <viennacl/linalg/row_scaling.hpp>

#ifdef VIENNACL_WITH_OPENCL
#include <viennacl/linalg/amg.hpp>
#include <viennacl/linalg/spai.hpp>
#endif

#define EXPORT_ICHOL0_PRECOND(MAT, TYPE)                                \
  bp::class_<vcl::linalg::ichol0_precond<vcl::MAT<TYPE> > >             \
  ("ichol0_precond_" #MAT "_" #TYPE,                                    \
   bp::init<vcl::MAT<TYPE> const&, vcl::linalg::ichol0_tag const&>())   \
  ;

#define EXPORT_ILUT_PRECOND(MAT, TYPE)                                  \
  bp::class_<vcl::linalg::ilut_precond<vcl::MAT<TYPE> > >               \
  ("ilut_precond_" #MAT "_" #TYPE,                                      \
   bp::init<vcl::MAT<TYPE> const&, vcl::linalg::ilut_tag const&>())     \
  ;

#define EXPORT_ILU0_PRECOND(MAT, TYPE)                                  \
  bp::class_<vcl::linalg::ilu0_precond<vcl::MAT<TYPE> > >               \
  ("ilu0_precond_" #MAT "_" #TYPE,                                      \
   bp::init<vcl::MAT<TYPE> const&, vcl::linalg::ilu0_tag const&>())     \
  ;

#define EXPORT_BLOCK_ILUT_PRECOND(MAT, TYPE)                            \
  bp::class_<vcl::linalg::block_ilu_precond<vcl::MAT<TYPE> , vcl::linalg::ilut_tag> > \
  ("block_ilut_precond_" #MAT "_" #TYPE,                                \
   bp::init<vcl::MAT<TYPE> const&, vcl::linalg::ilut_tag const&, vcl::vcl_size_t>()) \
  ;

#define EXPORT_BLOCK_ILU0_PRECOND(MAT, TYPE)                            \
  bp::class_<vcl::linalg::block_ilu_precond<vcl::MAT<TYPE>, vcl::linalg::ilu0_tag> > \
  ("block_ilu0_precond_" #MAT "_" #TYPE,                                \
   bp::init<vcl::MAT<TYPE> const&, vcl::linalg::ilu0_tag const&, vcl::vcl_size_t>()) \
  ;

#define EXPORT_JACOBI_PRECOND(MAT, TYPE)                                \
  bp::class_<vcl::linalg::jacobi_precond<vcl::MAT<TYPE> > >             \
  ("jacobi_precond_" #MAT "_" #TYPE,                                    \
   bp::init<vcl::MAT<TYPE> const&, vcl::linalg::jacobi_tag const&>())   \
  ;

#define EXPORT_ROW_SCALING_PRECOND(MAT, TYPE)                           \
  bp::class_<vcl::linalg::row_scaling<vcl::MAT<TYPE> > >                \
  ("row_scaling_precond_" #MAT "_" #TYPE,                               \
   bp::init<vcl::MAT<TYPE> const&, vcl::linalg::row_scaling_tag const&>()) \
  ;

#define EXPORT_AMG_PRECOND(MAT, TYPE)                                   \
  bp::class_<vcl::linalg::amg_precond<vcl::MAT<TYPE> > >                \
  ("amg_precond_" #MAT "_" #TYPE,                                       \
   bp::init<vcl::MAT<TYPE> const&, vcl::linalg::amg_tag const&>())      \
  ;

#define EXPORT_SPAI_PRECOND(MAT, TYPE)                                  \
  bp::class_<vcl::linalg::spai_precond<vcl::MAT<TYPE> > >               \
  ("spai_precond_" #MAT "_" #TYPE,                                      \
   bp::init<vcl::MAT<TYPE> const&, vcl::linalg::spai_tag const&>())     \
  ;

#define EXPORT_FSPAI_PRECOND(MAT, TYPE)                                 \
  bp::class_<vcl::linalg::fspai_precond<vcl::MAT<TYPE> > >              \
  ("fspai_precond_" #MAT "_" #TYPE,                                     \
   bp::init<vcl::MAT<TYPE> const&, vcl::linalg::fspai_tag const&>())    \
  ;

#endif
