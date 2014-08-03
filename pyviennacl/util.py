import pyviennacl as p
from . import _viennacl
from numpy import (ndarray, array,
                   result_type as np_result_type)
import logging

default_log_handler = logging.StreamHandler()
default_log_handler.setFormatter(logging.Formatter(
    "%(levelname)s %(asctime)s %(name)s %(lineno)d %(funcName)s\n  %(message)s"
))
logging.getLogger('pyviennacl').addHandler(default_log_handler)

def fix_operand(opand, node=None):
    """
    TODO docstring
    """
    if isinstance(opand, list):
        opand = from_ndarray(array(opand))
    if isinstance(opand, ndarray):
        return from_ndarray(opand)
    if (np_result_type(opand).name in p.HostScalarTypes
        and not isinstance(opand, p.MagicMethods)):
        return p.HostScalar(opand)
    if isinstance(opand, p.Node):
        if opand.flushed:
            return opand.result
        if opand.no_fix:
            return opand
        # TODO: REMOVE NEXT TEST
        if opand.operation_node_type_family == _viennacl.operation_node_type_family.OPERATION_UNARY_TYPE_FAMILY and not (type(opand) == p.Assign or type(node) == p.Assign):
            opand.no_fix = True
            return opand.result
    return opand


def from_ndarray(obj):
    """
    Convert a NumPy ``ndarray`` into a PyViennaCL object of appropriate
    dimensionality.

    Parameters
    ----------
    obj : array-like

    Returns
    -------
    new : {Vector, Matrix}
        ``Vector`` if ``obj`` has 1 dimension; ``Matrix`` if 2.

    Raises
    ------
    AttributeError
        If ``obj`` has less than 1 or more than 2 dimensions.
    """
    if obj.ndim == 1:
        new = p.Vector(obj)
    elif obj.ndim == 2:
        new = p.Matrix(obj)
    else:
        raise AttributeError("Cannot cope with %d dimensions!" % self.operands[0].ndim)
    return new
