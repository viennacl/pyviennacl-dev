#include "preconditioners.hpp"

PYVCL_SUBMODULE(preconditioners)
{

  /*

    TODO:
    + preconditioners themselves
      - do this in .hpp and export for the various sparse types
      - ilut_precond
      - ilu0_precond
      - block_ilu_precond (check docs for args; ilu0/ilut)
      - jacobi_precond
      - row_scaling_precond
    + add preconditioner support to solver calls

   */

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::ilut_tag, bool,
                                  use_level_scheduling,
                                  ilut_get_level_scheduling, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::ilut_tag, void,
                                  use_level_scheduling,
                                  ilut_set_level_scheduling, (bool))
    bp::class_<vcl::linalg::ilut_tag>("ilut_tag")
    .def(bp::init<unsigned int, double, bool>())
    .add_property("entries_per_row",
                  &vcl::linalg::ilut_tag::get_entries_per_row,
                  &vcl::linalg::ilut_tag::set_entries_per_row)
    .add_property("drop_tolerance",
                  &vcl::linalg::ilut_tag::get_drop_tolerance,
                  &vcl::linalg::ilut_tag::set_drop_tolerance)
    .add_property("use_level_scheduling",
                  ilut_get_level_scheduling,
                  ilut_set_level_scheduling)
    ;

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::ilu0_tag, bool,
                                  use_level_scheduling,
                                  ilu0_get_level_scheduling, () const)
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::ilu0_tag, void,
                                  use_level_scheduling,
                                  ilu0_set_level_scheduling, (bool))
    bp::class_<vcl::linalg::ilu0_tag>("ilu0_tag")
    .def(bp::init<bool>())
    .add_property("use_level_scheduling",
                  ilu0_get_level_scheduling,
                  ilu0_set_level_scheduling)
    ;
  
  bp::class_<vcl::linalg::jacobi_tag>("jacobi_tag");

  bp::class_<vcl::linalg::row_scaling_tag>("row_scaling_tag")
    .def(bp::init<unsigned int>())
    .add_property("norm", &vcl::linalg::row_scaling_tag::norm)
    ;

}
