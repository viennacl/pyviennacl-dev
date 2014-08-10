#include "common.hpp"

#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/linalg/norm_inf.hpp>
#include <viennacl/linalg/norm_frobenius.hpp>
#include <viennacl/linalg/qr.hpp>

#ifdef VIENNACL_WITH_OPENCL
#include <viennacl/linalg/nmf.hpp>
#include <viennacl/linalg/svd.hpp>
#include <viennacl/fft.hpp>
#else
// This is here to keep #ifdefs to a minimum
namespace viennacl {
  namespace linalg {
    struct nmf_config{};
  } 
}
#endif

DO_OP_FUNC(op_inner_prod)
{
  return vcl::linalg::inner_prod(o.operand1, o.operand2);
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_outer_prod)
{
  return vcl::linalg::outer_prod(o.operand1, o.operand2);
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_element_pow)
{
  return vcl::linalg::element_pow(o.operand1, o.operand2);
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_norm_1)
{
  return vcl::linalg::norm_1(o.operand1);
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_norm_2)
{
  return vcl::linalg::norm_2(o.operand1);
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_norm_inf)
{
  return vcl::linalg::norm_inf(o.operand1);
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_norm_frobenius)
{
  return vcl::linalg::norm_frobenius(o.operand1);
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_plane_rotation)
{
  vcl::linalg::plane_rotation(o.operand1, o.operand2,
			      o.operand3, o.operand4);
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_inplace_qr) {
  return vcl::linalg::inplace_qr(o.operand1, o.operand2);
}
CLOSE_OP_FUNC;


DO_OP_FUNC(op_inplace_qr_apply_trans_q) {
  vcl::linalg::inplace_qr_apply_trans_Q(o.operand1, o.operand2, o.operand3);
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_recoverq) {
  vcl::linalg::recoverQ(o.operand1, o.operand2, o.operand3, o.operand4);
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_fft_2d)
{
#ifdef VIENNACL_WITH_OPENCL
  vcl::fft(o.operand1, o.operand2, o.operand3, o.operand4);
#endif
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_fft_3d)
{
#ifdef VIENNACL_WITH_OPENCL
  vcl::fft(o.operand1, o.operand2, o.operand3);
#endif
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_inplace_fft_2d)
{
#ifdef VIENNACL_WITH_OPENCL
  vcl::inplace_fft(o.operand1, o.operand2, o.operand3);
#endif
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_inplace_fft_3d)
{
#ifdef VIENNACL_WITH_OPENCL
  vcl::inplace_fft(o.operand1, o.operand2);
#endif
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_ifft_2d)
{
#ifdef VIENNACL_WITH_OPENCL
  vcl::ifft(o.operand1, o.operand2, o.operand3);
#endif
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_inplace_ifft_2d)
{
#ifdef VIENNACL_WITH_OPENCL
  vcl::inplace_ifft(o.operand1, o.operand2);
#endif
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_convolve_2d)
{
#ifdef VIENNACL_WITH_OPENCL
  vcl::linalg::convolve(o.operand1, o.operand2, o.operand3);
#endif
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_convolve_i_2d)
{
#ifdef VIENNACL_WITH_OPENCL
  vcl::linalg::convolve_i(o.operand1, o.operand2, o.operand3);
#endif
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_fft_normalize_2d)
{
#ifdef VIENNACL_WITH_OPENCL
  vcl::detail::fft::normalize(o.operand1);
#endif
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_nmf) {
#ifdef VIENNACL_WITH_OPENCL
  vcl::linalg::nmf(o.operand1, o.operand2, o.operand3, o.operand4);
#endif
  return bp::object();
}
CLOSE_OP_FUNC;

DO_OP_FUNC(op_svd) {
#ifdef VIENNACL_WITH_OPENCL
  vcl::linalg::svd(o.operand1, o.operand2, o.operand3);
#endif
  return bp::object();
}
CLOSE_OP_FUNC;

#define EXPORT_FUNCTIONS_F(TYPE, F)                                     \
  bp::def("outer", pyvcl_do_2ary_op<vcl::matrix<TYPE, vcl::column_major>, \
          vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          op_outer_prod>);                                              \
  bp::def("element_pow", pyvcl_do_2ary_op<vcl::matrix<TYPE, F>,         \
          vcl::matrix_base<TYPE>&, vcl::matrix_base<TYPE>&,             \
          op_element_pow>);                                             \
  bp::def("norm_frobenius", pyvcl_do_1ary_op<vcl::scalar<TYPE>,         \
          vcl::matrix<TYPE, F>&,                                        \
          op_norm_frobenius>);                                          \
  bp::def("inplace_qr", pyvcl_do_2ary_op<std::vector<TYPE>,             \
          vcl::matrix<TYPE, F>&, vcl::vcl_size_t,                       \
          op_inplace_qr>);                                              \
  bp::def("inplace_qr_apply_trans_Q", pyvcl_do_3ary_op<bp::object,      \
          const vcl::matrix<TYPE, F>&, const std::vector<TYPE>&,        \
          vcl::vector<TYPE>&,                                           \
          op_inplace_qr_apply_trans_q>);                                \
  bp::def("recoverQ", pyvcl_do_4ary_op<bp::object,                      \
          const vcl::matrix<TYPE, F>&, const std::vector<TYPE>&,        \
          vcl::matrix<TYPE, F>&, vcl::matrix<TYPE, F>&,                 \
          op_recoverq>);

// TODO: NMF, FFT, SVD only support row_major right now
#define EXPORT_FUNCTIONS(TYPE)                                          \
  EXPORT_FUNCTIONS_F(TYPE, vcl::row_major);                             \
  EXPORT_FUNCTIONS_F(TYPE, vcl::column_major);                          \
  bp::def("nmf", pyvcl_do_4ary_op<bp::object,                           \
          const vcl::matrix<TYPE, vcl::row_major>&,                     \
          vcl::matrix<TYPE, vcl::row_major>&,                           \
          vcl::matrix<TYPE, vcl::row_major>&,                           \
          const vcl::linalg::nmf_config&,                               \
          op_nmf>);                                                     \
  bp::def("svd", pyvcl_do_3ary_op<bp::object,                           \
          vcl::matrix<TYPE, vcl::row_major>&,                           \
          vcl::matrix<TYPE, vcl::row_major>&,                           \
          vcl::matrix<TYPE, vcl::row_major>&,                           \
          op_svd>);                                                     \
  bp::def("inner_prod", pyvcl_do_2ary_op<vcl::scalar<TYPE>,             \
          vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          op_inner_prod>);                                              \
  bp::def("element_pow", pyvcl_do_2ary_op<vcl::vector<TYPE>,            \
          vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          op_element_pow>);                                             \
  bp::def("plane_rotation", pyvcl_do_4ary_op<bp::object,                \
          vcl::vector_base<TYPE>&, vcl::vector_base<TYPE>&,             \
          TYPE, TYPE,                                                   \
          op_plane_rotation>);                                          \
  bp::def("norm_1", pyvcl_do_1ary_op<vcl::scalar<TYPE>,                 \
          vcl::vector_base<TYPE>&,                                      \
          op_norm_1>);                                                  \
  bp::def("norm_2", pyvcl_do_1ary_op<vcl::scalar<TYPE>,                 \
          vcl::vector_base<TYPE>&,                                      \
          op_norm_2>);                                                  \
  bp::def("norm_inf", pyvcl_do_1ary_op<vcl::scalar<TYPE>,               \
          vcl::vector_base<TYPE>&,                                      \
          op_norm_inf>);                                                \
  bp::def("fft", pyvcl_do_4ary_op<bp::object,                           \
          vcl::vector<TYPE>&, vcl::vector<TYPE>&,                       \
          vcl::vcl_size_t, TYPE,                                        \
          op_fft_2d>);                                                  \
  bp::def("ifft", pyvcl_do_3ary_op<bp::object,                          \
          vcl::vector<TYPE>&, vcl::vector<TYPE>&, vcl::vcl_size_t,      \
          op_ifft_2d>);                                                 \
  bp::def("inplace_fft", pyvcl_do_3ary_op<bp::object,                   \
          vcl::vector<TYPE>&, vcl::vcl_size_t, TYPE,                    \
          op_inplace_fft_2d>);                                          \
  bp::def("inplace_ifft", pyvcl_do_2ary_op<bp::object,                  \
          vcl::vector<TYPE>&, vcl::vcl_size_t,                          \
          op_inplace_ifft_2d>);                                         \
  bp::def("fft", pyvcl_do_3ary_op<bp::object,                           \
          vcl::matrix<TYPE, vcl::row_major>&,                           \
          vcl::matrix<TYPE, vcl::row_major>&, TYPE,                     \
          op_fft_3d>);                                                  \
  bp::def("inplace_fft", pyvcl_do_2ary_op<bp::object,                   \
          vcl::matrix<TYPE, vcl::row_major>&, TYPE,                     \
          op_inplace_fft_3d>);                                          \
  bp::def("convolve", pyvcl_do_3ary_op<bp::object,                      \
          vcl::vector<TYPE>&, vcl::vector<TYPE>&, vcl::vector<TYPE>&,   \
          op_convolve_2d>);                                             \
  bp::def("convolve_i", pyvcl_do_3ary_op<bp::object,                    \
          vcl::vector<TYPE>&, vcl::vector<TYPE>&, vcl::vector<TYPE>&,   \
          op_convolve_i_2d>);                                           \
  bp::def("normalize", pyvcl_do_1ary_op<bp::object, vcl::vector<TYPE>&, \
          op_fft_normalize_2d>);


PYVCL_SUBMODULE(extra_functions)
{

  bp::def("backend_finish", vcl::backend::finish);

  bp::class_<vcl::range>("range",
                         bp::init<vcl::vcl_size_t, vcl::vcl_size_t>());
  bp::class_<vcl::slice>("slice",
                         bp::init<vcl::vcl_size_t, vcl::vcl_size_t, vcl::vcl_size_t>());

#ifdef VIENNACL_WITH_OPENCL
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::nmf_config, double,
                                  tolerance, get_tolerance, () const);
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::nmf_config, void,
                                  tolerance, set_tolerance, (double));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::nmf_config, double,
                                  stagnation_tolerance, 
                                  get_stagnation_tolerance,
                                  () const);
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::nmf_config, void,
                                  stagnation_tolerance, 
                                  set_stagnation_tolerance,
                                  (double));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::nmf_config, vcl::vcl_size_t,
                                  max_iterations, 
                                  get_max_iterations,
                                  () const);
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::nmf_config, vcl::vcl_size_t,
                                  iters,
                                  get_iterations,
                                  () const);
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::nmf_config, void,
                                  max_iterations, 
                                  set_max_iterations,
                                  (vcl::vcl_size_t));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::nmf_config, vcl::vcl_size_t,
                                  check_after_steps, 
                                  get_check_after_steps,
                                  () const);
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::nmf_config, void,
                                  check_after_steps, 
                                  set_check_after_steps,
                                  (vcl::vcl_size_t));
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::nmf_config, bool,
                                  print_relative_error, 
                                  get_print_relative_error,
                                  () const);
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::nmf_config, void,
                                  print_relative_error, 
                                  set_print_relative_error,
                                  (bool));
  bp::class_<vcl::linalg::nmf_config>
    ("nmf_config",
     bp::init<double, double, vcl::vcl_size_t, vcl::vcl_size_t>())
    .add_property("iterations", get_iterations)
    .add_property("tolerance", get_tolerance, set_tolerance)
    .add_property("stagnation_tolerance",
                  get_stagnation_tolerance, set_stagnation_tolerance)
    .add_property("max_iterations",
                  get_max_iterations, set_max_iterations)
    .add_property("check_after_steps",
                  get_check_after_steps, set_check_after_steps)
    .add_property("print_relative_error",
                  get_print_relative_error, set_print_relative_error)
    ;
#else
  bp::class_<vcl::linalg::nmf_config>("nmf_config", bp::no_init);
#endif

  /* TODO missing: char, short, uchar, ushort
     Some of these only make compile on Windows for float types -- eg norm_2, which
       tries to do a sqrt on a long without converting it to some float type.
  EXPORT_FUNCTIONS(int);
  EXPORT_FUNCTIONS(long);
  EXPORT_FUNCTIONS(uint);
  EXPORT_FUNCTIONS(ulong);
  */
  EXPORT_FUNCTIONS(double);
  EXPORT_FUNCTIONS(float);

}
