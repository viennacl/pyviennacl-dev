.. PyViennaCL documentation master file, created by
   sphinx-quickstart on Tue Sep 17 16:16:55 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: pyviennacl

PyViennaCL: easy, powerful scientific computing
===============================================

.. toctree::
   :titlesonly:
   :numbered: 2

   README
   pycore
   linalg
   tags
   vclmath
   backend
   opencl
   io
   util
   device_specific

Overview
========

.. note::

   All of PyViennaCL's functionality is available from the top-level
   :mod:`pyviennacl` namespace.

This extension provides the Python bindings for the `ViennaCL
<http://viennacl.sourceforge.net/>`_ linear algebra and numerical
computation library for GPGPU and heterogeneous systems. `ViennaCL
<http://viennacl.sourceforge.net/>`_ itself is a header-only C++
library, so these bindings make available to Python programmers
ViennaCL's fast OpenCL and CUDA algorithms, in a way that is idiomatic
and compatible with the Python community's most popular scientific
packages, NumPy and SciPy.

The PyViennaCL public API is divided into nine submodules, as
described by the documentation linked above, and code examples given
below.

Since all of PyViennaCL's functionality is available from the
top-level pyviennacl namespace, if you want help on the
:class:`Matrix` class, you can just run::

  >>> import pyviennacl as p
  >>> help(p.Matrix)                                     # doctest: +SKIP

For help on PyViennaCL's core functionality in general, or
PyViennaCL's high-level linear algebra functions, run::

   >>> help(p.pycore)                                    # doctest: +SKIP

or::

   >>> help(p.linalg)                                    # doctest: +SKIP


Usage examples
==============

.. toctree::
   :maxdepth: 2

   examples/index.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

