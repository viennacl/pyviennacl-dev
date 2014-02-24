#ifndef _PYVIENNACL_VECTOR_H
#define _PYVIENNACL_VECTOR_H

#include "viennacl.h"

#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>

#include <viennacl/linalg/sparse_matrix_operations.hpp>
#include <viennacl/compressed_matrix.hpp>
#include <viennacl/coordinate_matrix.hpp>
#include <viennacl/ell_matrix.hpp>
#include <viennacl/hyb_matrix.hpp>

namespace ublas = boost::numeric::ublas;

template <class ScalarType>
class cpu_compressed_matrix_wrapper
{
  typedef ublas::compressed_matrix<ScalarType, ublas::row_major> ublas_sparse_t;
  ublas_sparse_t cpu_compressed_matrix;
  bool _dirty;
  bp::list* _places;

public:

  void update_places()
  {

    if (!_dirty)
      return;

    typedef typename ublas_sparse_t::iterator1 it1;
    typedef typename ublas_sparse_t::iterator2 it2;

    if (_places)
      delete _places;

    _places = new bp::list;

    for (it1 i = cpu_compressed_matrix.begin1();
         i != cpu_compressed_matrix.end1(); ++i) {
      for (it2 j = i.begin(); j != i.end(); ++j) {

	if (cpu_compressed_matrix(j.index1(), j.index2())) {
          _places->append(bp::make_tuple(j.index1(), j.index2()));
        }

      }
    }

    _dirty = false;

  }

  bp::list places() 
  {
    if (_dirty)
      update_places();

    return *_places;
  }

  vcl::vcl_size_t nnz()
  {
    if (_dirty)
      update_places();

    return bp::len(*_places);
  }

  cpu_compressed_matrix_wrapper()
  {
    _places = NULL;
    cpu_compressed_matrix = ublas_sparse_t(0,0,0);
  }

  cpu_compressed_matrix_wrapper(vcl::vcl_size_t _size1, vcl::vcl_size_t _size2)
  {
    _places = NULL;
    cpu_compressed_matrix = ublas_sparse_t(_size1, _size2);
  }

  cpu_compressed_matrix_wrapper(vcl::vcl_size_t _size1, vcl::vcl_size_t _size2, vcl::vcl_size_t _nnz)
  {
    _places = NULL;
    cpu_compressed_matrix = ublas_sparse_t(_size1, _size2, _nnz);
  }

  cpu_compressed_matrix_wrapper(const cpu_compressed_matrix_wrapper& w)
    : cpu_compressed_matrix(w.cpu_compressed_matrix)
  { 
    _places = NULL;
    _dirty = true;
  }

  template<class SparseT>
  cpu_compressed_matrix_wrapper(const SparseT& vcl_sparse_matrix)
  {
    cpu_compressed_matrix = ublas_sparse_t(vcl_sparse_matrix.size1(),
                                           vcl_sparse_matrix.size2());
    vcl::copy(vcl_sparse_matrix, cpu_compressed_matrix);
    _places = NULL;
    _dirty = true;
  }

  cpu_compressed_matrix_wrapper(const np::ndarray& array)
  {
    _places = NULL;

    int d = array.get_nd();
    if (d != 2) {
      PyErr_SetString(PyExc_TypeError, "Can only create a matrix from a 2-D array!");
      bp::throw_error_already_set();
    }
    
    vcl::vcl_size_t n = array.shape(0);
    vcl::vcl_size_t m = array.shape(1);
    
    cpu_compressed_matrix = ublas_sparse_t(n, m);
    
    for (vcl::vcl_size_t i = 0; i < n; ++i) {
      for (vcl::vcl_size_t j = 0; j < m; ++j) {
	ScalarType val = bp::extract<ScalarType>(array[i][j]);
	if (val != 0)
          set_entry(i, j, val);
      }
    }
  }

  np::ndarray as_ndarray()
  {
    np::dtype dt = np::dtype::get_builtin<ScalarType>();
    bp::tuple shape = bp::make_tuple(size1(), size2());
    
    np::ndarray array = np::zeros(shape, dt);

    typedef typename ublas_sparse_t::iterator1 it1;
    typedef typename ublas_sparse_t::iterator2 it2;

    for (it1 i = cpu_compressed_matrix.begin1();
         i != cpu_compressed_matrix.end1(); ++i) {
      for (it2 j = i.begin(); j != i.end(); ++j) {

	if (cpu_compressed_matrix(j.index1(), j.index2()) != 0) {
          array[j.index1()][j.index2()] = get_entry(j.index1(), j.index2());
        }

      }
    }

    return array;
  }

  template<class SparseT>
  vcl::tools::shared_ptr<SparseT>
  as_vcl_sparse_matrix()
  {
    SparseT* vcl_sparse_matrix = new SparseT();
    vcl::copy(cpu_compressed_matrix, *vcl_sparse_matrix);
    return vcl::tools::shared_ptr<SparseT>(vcl_sparse_matrix);
  }

  template<class SparseT>
  vcl::tools::shared_ptr<SparseT>
  as_vcl_sparse_matrix_with_size()
  {
    SparseT* vcl_sparse_matrix = new SparseT(size1(), size2(), nnz());
    vcl::copy(cpu_compressed_matrix, *vcl_sparse_matrix);
    return vcl::tools::shared_ptr<SparseT>(vcl_sparse_matrix);
  }

  vcl::vcl_size_t size1() const
  {
    return cpu_compressed_matrix.size1();
  }

  vcl::vcl_size_t size2() const
  {
    return cpu_compressed_matrix.size2();
  }

  void resize(vcl::vcl_size_t _size1, vcl::vcl_size_t _size2)
  {

    if ((_size1 == size1()) && (_size2 == size2()))
      return;

    // TODO NB: ublas compressed matrix does not support preserve on resize
    //          so this below is annoyingly hacky...

    ublas_sparse_t temp(cpu_compressed_matrix); // Incurs a copy of all the data!!
    cpu_compressed_matrix.resize(_size1, _size2, false); // preserve == false!

    /*
    if (_places)
      delete _places;

    _places = new bp::list;
    */

    typedef typename ublas_sparse_t::iterator1 it1;
    typedef typename ublas_sparse_t::iterator2 it2;

    for (it1 i = temp.begin1(); i != temp.end1(); ++i) {
      for (it2 j = i.begin(); j != i.end(); ++j) {
	if ((temp(j.index1(), j.index2()) != 0)
            && (j.index1() < _size1) && (j.index2() < _size2)) {
          cpu_compressed_matrix(j.index1(), j.index2()) = temp(j.index1(), j.index2());
          //_places->append(bp::make_tuple(j.index1(), j.index2()));
        }
      }
    }

    _dirty = true; //false;

  }
  
  void set_entry(vcl::vcl_size_t n, vcl::vcl_size_t m, ScalarType val) 
  {
    if (n >= size1()) {
      if (m >= size2())
        resize(n+1, m+1);
      else
        resize(n+1, size2());
    } else {
      if (m >= size2())
        resize(size1(), m+1);
    }

    ScalarType old = cpu_compressed_matrix(n, m);
    if (val != old) {
      cpu_compressed_matrix(n, m) = val;
      _dirty = true;
    }
  }

  // Need this because bp cannot deal with operator()
  ScalarType get_entry(vcl::vcl_size_t n, vcl::vcl_size_t m) const
  {
    return cpu_compressed_matrix(n, m);
  }

};

#endif
