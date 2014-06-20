#include "iterative_solvers.hpp"

PYVCL_SUBMODULE(iterative_solvers)
{

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::cg_tag, unsigned int,
                                  iters, get_cg_iters, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::cg_tag, void,
                                  iters, set_cg_iters, (unsigned int) const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::cg_tag, double,
                                  error, get_cg_error, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::cg_tag, void,
                                  error, set_cg_error, (double) const)
  bp::class_<vcl::linalg::cg_tag>("cg_tag")
    .def(bp::init<double, vcl::vcl_size_t>())
    .add_property("tolerance", &vcl::linalg::cg_tag::tolerance)
    .add_property("max_iterations", &vcl::linalg::cg_tag::max_iterations)
    .add_property("iters", get_cg_iters, set_cg_iters)
    .add_property("error", get_cg_error, set_cg_error)
    ;

#ifdef VIENNACL_WITH_OPENCL
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::mixed_precision_cg_tag,
                                  unsigned int,
                                  iters, get_mixed_precision_cg_iters,
                                  () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::mixed_precision_cg_tag,
                                  void,
                                  iters, set_mixed_precision_cg_iters,
                                  (unsigned int) const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::mixed_precision_cg_tag,
                                  double,
                                  error, get_mixed_precision_cg_error,
                                  () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::mixed_precision_cg_tag,
                                  void,
                                  error, set_mixed_precision_cg_error,
                                  (double) const)
  bp::class_<vcl::linalg::mixed_precision_cg_tag>
    ("mixed_precision_cg_tag",
     bp::init<double, unsigned int, float>())
    .add_property("tolerance",
                  &vcl::linalg::mixed_precision_cg_tag::tolerance)
    .add_property("inner_tolerance",
                  &vcl::linalg::mixed_precision_cg_tag::inner_tolerance)
    .add_property("max_iterations",
                  &vcl::linalg::mixed_precision_cg_tag::max_iterations)
    .add_property("iters",
                  get_mixed_precision_cg_iters, set_mixed_precision_cg_iters)
    .add_property("error",
                  get_mixed_precision_cg_error, set_mixed_precision_cg_error)
    ;
#endif

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::bicgstab_tag, vcl::vcl_size_t,
                                  iters, get_bicgstab_iters, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::bicgstab_tag, void,
                                  iters, set_bicgstab_iters,
                                  (vcl::vcl_size_t) const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::bicgstab_tag, double,
                                  error, get_bicgstab_error, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::bicgstab_tag, void,
                                  error, set_bicgstab_error, (double) const)
  bp::class_<vcl::linalg::bicgstab_tag>("bicgstab_tag")
    .def(bp::init<double, vcl::vcl_size_t, vcl::vcl_size_t>())
    .add_property("tolerance", &vcl::linalg::bicgstab_tag::tolerance)
    .add_property("max_iterations",
                  &vcl::linalg::bicgstab_tag::max_iterations)
    .add_property("max_iterations_before_restart",
                  &vcl::linalg::bicgstab_tag::max_iterations_before_restart)
    .add_property("iters", get_bicgstab_iters, set_bicgstab_iters)
    .add_property("error", get_bicgstab_error, set_bicgstab_error)
    ;

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::gmres_tag, unsigned int,
                                  iters, get_gmres_iters, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::gmres_tag, void,
                                  iters, set_gmres_iters, (unsigned int) const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::gmres_tag, double,
                                  error, get_gmres_error, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::gmres_tag, void,
                                  error, set_gmres_error, (double) const)
  bp::class_<vcl::linalg::gmres_tag>("gmres_tag")
    .def(bp::init<double, vcl::vcl_size_t, vcl::vcl_size_t>())
    .add_property("tolerance", &vcl::linalg::gmres_tag::tolerance)
    .add_property("max_iterations", &vcl::linalg::gmres_tag::max_iterations)
    .add_property("iters", get_gmres_iters, set_gmres_iters)
    .add_property("error", get_gmres_error, set_gmres_error)
    .add_property("krylov_dim", &vcl::linalg::gmres_tag::krylov_dim)
    .add_property("max_restarts", &vcl::linalg::gmres_tag::max_restarts)
    ;

  /*
  EXPORT_DENSE_ITERATIVE_SOLVERS(float);
  EXPORT_DENSE_ITERATIVE_SOLVERS(double);
  /...* These don't really make sense:
  EXPORT_DENSE_ITERATIVE_SOLVERS(int);
  EXPORT_DENSE_ITERATIVE_SOLVERS(uint);
  EXPORT_DENSE_ITERATIVE_SOLVERS(long);
  EXPORT_DENSE_ITERATIVE_SOLVERS(ulong);
  */

  EXPORT_SPARSE_ITERATIVE_SOLVERS(float);
  EXPORT_SPARSE_ITERATIVE_SOLVERS(double);

}

