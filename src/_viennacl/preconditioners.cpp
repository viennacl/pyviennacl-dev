#include "preconditioners.hpp"

PYVCL_SUBMODULE(preconditioners)
{

  /*
  
    TODO:
    + implement python layer

   */


  bp::class_<vcl::linalg::no_precond>("no_precond");

  // ICHOL0

  bp::class_<vcl::linalg::ichol0_tag>("ichol0_tag");

  EXPORT_ICHOL0_PRECOND(vcl::compressed_matrix<float>);
  EXPORT_ICHOL0_PRECOND(vcl::compressed_matrix<double>);

  // [BLOCK-]ILU(T/0)

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::ilut_tag, bool,
                                  use_level_scheduling,
                                  ilut_get_level_scheduling, () const);
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::ilut_tag, void,
                                  use_level_scheduling,
                                  ilut_set_level_scheduling, (bool));
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

  EXPORT_ILUT_PRECOND(vcl::compressed_matrix<float>);
  EXPORT_ILUT_PRECOND(vcl::compressed_matrix<double>);

  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::ilu0_tag, bool,
                                  use_level_scheduling,
                                  ilu0_get_level_scheduling, () const);
  DISAMBIGUATE_CLASS_FUNCTION_PTR(vcl::linalg::ilu0_tag, void,
                                  use_level_scheduling,
                                  ilu0_set_level_scheduling, (bool));
  bp::class_<vcl::linalg::ilu0_tag>("ilu0_tag")
    .def(bp::init<bool>())
    .add_property("use_level_scheduling",
                  ilu0_get_level_scheduling,
                  ilu0_set_level_scheduling)
    ;


  EXPORT_ILU0_PRECOND(vcl::compressed_matrix<float>);
  EXPORT_ILU0_PRECOND(vcl::compressed_matrix<double>);

  EXPORT_BLOCK_ILUT_PRECOND(vcl::compressed_matrix<float>);
  EXPORT_BLOCK_ILUT_PRECOND(vcl::compressed_matrix<double>);
  EXPORT_BLOCK_ILU0_PRECOND(vcl::compressed_matrix<float>);
  EXPORT_BLOCK_ILU0_PRECOND(vcl::compressed_matrix<double>);
  
  // JACOBI

  bp::class_<vcl::linalg::jacobi_tag>("jacobi_tag");

  EXPORT_JACOBI_PRECOND(vcl::compressed_matrix<float>);
  EXPORT_JACOBI_PRECOND(vcl::compressed_matrix<double>);
  EXPORT_JACOBI_PRECOND(vcl::coordinate_matrix<float>);
  EXPORT_JACOBI_PRECOND(vcl::coordinate_matrix<double>);

  // ROW SCALING

  bp::class_<vcl::linalg::row_scaling_tag>("row_scaling_tag")
    .def(bp::init<unsigned int>())
    .add_property("norm", &vcl::linalg::row_scaling_tag::norm)
    ;

  EXPORT_ROW_SCALING_PRECOND(vcl::compressed_matrix<float>);
  EXPORT_ROW_SCALING_PRECOND(vcl::compressed_matrix<double>);
  EXPORT_ROW_SCALING_PRECOND(vcl::coordinate_matrix<float>);
  EXPORT_ROW_SCALING_PRECOND(vcl::coordinate_matrix<double>);

#ifdef VIENNACL_WITH_OPENCL

  // AMG

  bp::scope().attr("VIENNACL_AMG_COARSE_RS") = VIENNACL_AMG_COARSE_RS;
  bp::scope().attr("VIENNACL_AMG_COARSE_ONEPASS") = VIENNACL_AMG_COARSE_ONEPASS;
  bp::scope().attr("VIENNACL_AMG_COARSE_RS0") = VIENNACL_AMG_COARSE_RS0;
  bp::scope().attr("VIENNACL_AMG_COARSE_RS3") = VIENNACL_AMG_COARSE_RS3;
  bp::scope().attr("VIENNACL_AMG_COARSE_AG") = VIENNACL_AMG_COARSE_AG;

  bp::scope().attr("VIENNACL_AMG_INTERPOL_DIRECT") = VIENNACL_AMG_INTERPOL_DIRECT;
  bp::scope().attr("VIENNACL_AMG_INTERPOL_CLASSIC") = VIENNACL_AMG_INTERPOL_CLASSIC;
  bp::scope().attr("VIENNACL_AMG_INTERPOL_AG") = VIENNACL_AMG_INTERPOL_AG;
  bp::scope().attr("VIENNACL_AMG_INTERPOL_SA") = VIENNACL_AMG_INTERPOL_SA;

  bp::class_<vcl::linalg::amg_tag>("amg_tag")
    .def(bp::init<unsigned int, unsigned int,
         double, double, double, 
         unsigned int, unsigned int, unsigned int>())
    .add_property("coarse",
                  &vcl::linalg::amg_tag::get_coarse,
                  &vcl::linalg::amg_tag::set_coarse)
    .add_property("interpol",
                  &vcl::linalg::amg_tag::get_interpol,
                  &vcl::linalg::amg_tag::set_interpol)
    .add_property("threshold",
                  &vcl::linalg::amg_tag::get_threshold,
                  &vcl::linalg::amg_tag::set_threshold)
    .add_property("jacobiweight",
                  &vcl::linalg::amg_tag::get_jacobiweight,
                  &vcl::linalg::amg_tag::set_as)
    .add_property("interpolweight",
                  &vcl::linalg::amg_tag::get_interpolweight,
                  &vcl::linalg::amg_tag::set_interpolweight)
    .add_property("presmooth",
                  &vcl::linalg::amg_tag::get_presmooth,
                  &vcl::linalg::amg_tag::set_presmooth)
    .add_property("postsmooth",
                  &vcl::linalg::amg_tag::get_postsmooth,
                  &vcl::linalg::amg_tag::set_postsmooth)
    .add_property("coarselevels",
                  &vcl::linalg::amg_tag::get_coarselevels,
                  &vcl::linalg::amg_tag::set_coarselevels)
    ;

  EXPORT_AMG_PRECOND(vcl::compressed_matrix<float>);
  EXPORT_AMG_PRECOND(vcl::compressed_matrix<double>);

  // (F)SPAI

  bp::class_<vcl::linalg::spai_tag>("spai_tag")
    .def(bp::init<double, unsigned int, double, bool, bool>())
    .add_property("residual_norm_threshold",
                  &vcl::linalg::spai_tag::getResidualNormThreshold,
                  &vcl::linalg::spai_tag::setResidualNormThreshold)
    .add_property("iteration_limit",
                  &vcl::linalg::spai_tag::getIterationLimit,
                  &vcl::linalg::spai_tag::setIterationLimit)
    .add_property("residual_threshold",
                  &vcl::linalg::spai_tag::getResidualThreshold,
                  &vcl::linalg::spai_tag::setResidualThreshold)
    .add_property("is_static",
                  &vcl::linalg::spai_tag::getIsStatic,
                  &vcl::linalg::spai_tag::setIsStatic)
    .add_property("is_right",
                  &vcl::linalg::spai_tag::getIsRight,
                  &vcl::linalg::spai_tag::setIsRight)
    ;

  EXPORT_SPAI_PRECOND(vcl::compressed_matrix<float>);
  EXPORT_SPAI_PRECOND(vcl::compressed_matrix<double>);

  bp::class_<vcl::linalg::fspai_tag>("fspai_tag")
    .def(bp::init<double, unsigned int, bool, bool>())
    .add_property("residual_norm_threshold",
                  &vcl::linalg::fspai_tag::getResidualNormThreshold,
                  &vcl::linalg::fspai_tag::setResidualNormThreshold)
    .add_property("iteration_limit",
                  &vcl::linalg::fspai_tag::getIterationLimit,
                  &vcl::linalg::fspai_tag::setIterationLimit)
    .add_property("is_static",
                  &vcl::linalg::fspai_tag::getIsStatic,
                  &vcl::linalg::fspai_tag::setIsStatic)
    .add_property("is_right",
                  &vcl::linalg::fspai_tag::getIsRight,
                  &vcl::linalg::fspai_tag::setIsRight)
    ;

  EXPORT_FSPAI_PRECOND(vcl::compressed_matrix<float>);
  EXPORT_FSPAI_PRECOND(vcl::compressed_matrix<double>);

#endif // VIENNACL_WITH_OPENCL

}
