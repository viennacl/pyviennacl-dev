from . import _viennacl
import logging

default_log_handler = logging.StreamHandler()
default_log_handler.setFormatter(logging.Formatter(
    "%(levelname)s %(asctime)s %(name)s %(lineno)d %(funcName)s\n  %(message)s"
))
logging.getLogger('pyviennacl').addHandler(default_log_handler)

def fix_operand(opand):
    """
    If opand is a scalar type, wrap it in a PyViennaCL scalar class.
    """
    if isinstance(opand, list):
        opand = array(opand)
    if (np_result_type(opand).name in HostScalarTypes
        and not (isinstance(opand, MagicMethods)
                 or isinstance(opand, ndarray))):
        return HostScalar(opand)
    else: return opand

def backend_finish():
    """
    Block until any computation active on the compute backend is finished.
    """
    return _viennacl.backend_finish()

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
        new = Vector(obj)
    elif obj.ndim == 2:
        new = Matrix(obj)
    else:
        raise AttributeError("Cannot cope with %d dimensions!" % self.operands[0].ndim)
    return new

