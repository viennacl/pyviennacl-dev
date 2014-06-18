#ifndef _PYVIENNACL_PRECONDITIONERS_HPP
#define _PYVIENNACL_PRECONDITIONERS_HPP

#include "viennacl.h"
#include "sparse_matrix.h"

#include <viennacl/linalg/amg.hpp>
#include <viennacl/linalg/ichol.hpp>
#include <viennacl/linalg/ilu.hpp>
#include <viennacl/linalg/jacobi_precond.hpp>
#include <viennacl/linalg/row_scaling.hpp>
#include <viennacl/linalg/spai.hpp>

#define EXPORT_ICHOL0_PRECOND(MAT)                                      \
  bp::class_<vcl::linalg::ichol0_precond<MAT> >                         \
  ("ichol0_precond_" #MAT,                                              \
   bp::init<MAT const&, vcl::linalg::ichol0_tag const&>())              \
  ;

#define EXPORT_ILUT_PRECOND(MAT)                                        \
  bp::class_<vcl::linalg::ilut_precond<MAT> >                           \
  ("ilut_precond_" #MAT,                                                \
   bp::init<MAT const&, vcl::linalg::ilut_tag const&>())                \
  ;

#define EXPORT_ILU0_PRECOND(MAT)                                        \
  bp::class_<vcl::linalg::ilu0_precond<MAT> >                           \
  ("ilu0_precond_" #MAT,                                                \
   bp::init<MAT const&, vcl::linalg::ilu0_tag const&>())                \
  ;

#define EXPORT_BLOCK_ILUT_PRECOND(MAT)                                  \
  bp::class_<vcl::linalg::block_ilu_precond<MAT, vcl::linalg::ilut_tag> > \
  ("block_ilut_precond_" #MAT,                                          \
   bp::init<MAT const&, vcl::linalg::ilut_tag const&, vcl::vcl_size_t>()) \
  ;

#define EXPORT_BLOCK_ILU0_PRECOND(MAT)                                  \
  bp::class_<vcl::linalg::block_ilu_precond<MAT, vcl::linalg::ilu0_tag> > \
  ("block_ilu0_precond_" #MAT,                                          \
   bp::init<MAT const&, vcl::linalg::ilu0_tag const&, vcl::vcl_size_t>()) \
  ;

#define EXPORT_JACOBI_PRECOND(MAT)                                      \
  bp::class_<vcl::linalg::jacobi_precond<MAT> >                         \
  ("jacobi_precond_" #MAT,                                              \
   bp::init<MAT const&, vcl::linalg::jacobi_tag const&>())              \
  ;

#define EXPORT_ROW_SCALING_PRECOND(MAT)                                 \
  bp::class_<vcl::linalg::row_scaling<MAT> >                            \
  ("row_scaling_precond_" #MAT,                                         \
   bp::init<MAT const&, vcl::linalg::row_scaling_tag const&>())         \
  ;

#define EXPORT_AMG_PRECOND(MAT)                                         \
  bp::class_<vcl::linalg::amg_precond<MAT> >                            \
  ("amg_precond_" #MAT,                                                 \
   bp::init<MAT const&, vcl::linalg::amg_tag const&>())                 \
  ;

#define EXPORT_SPAI_PRECOND(MAT)                                        \
  bp::class_<vcl::linalg::spai_precond<MAT> >                           \
  ("spai_precond_" #MAT,                                                \
   bp::init<MAT const&, vcl::linalg::spai_tag const&>())                \
  ;

#define EXPORT_FSPAI_PRECOND(MAT)                                       \
  bp::class_<vcl::linalg::fspai_precond<MAT> >                          \
  ("fspai_precond_" #MAT,                                               \
   bp::init<MAT const&, vcl::linalg::fspai_tag const&>())               \
  ;

#endif
