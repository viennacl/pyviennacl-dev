import abc, logging
from . import _viennacl as _v
from .pycore import Node, Statement

class OrderType(object):
    def __init__(*args):
        raise TypeError("This class is not supposed to be instantiated")

class SequentialOrder(OrderType):
    vcl_order = _v.statements_tuple_order_type.SEQUENTIAL

class IndependentOrder(OrderType):
    vcl_order = _v.statements_tuple_order_type.INDEPENDENT


class StatementsTuple(object):
    vcl_statements_tuple = None

    def __init__(self, statements, order):
        def to_vcl_statement(s):
            if isinstance(s, Node):
                return Statement(s).vcl_statement
            else:
                return s.vcl_statement
        vcl_statements = list(map(to_vcl_statement, statements))
        self.order = order
        self.vcl_tuple = _v.statements_tuple(vcl_statements, order.vcl_order)


class TemplateBase(object):
    
    class Parameters(object):        
        
        def __init__(self, simd_width, local_sizes):
            self.simd_width = simd_width
            if len(local_sizes) >=1:
                self.local_size_0 = local_sizes[0]
            if len(local_sizes) >=2:
                self.local_size_1 = local_sizes[1]
        
        def vcl_arguments_order():
            return

        def make_parameters(self):
            return [self.__dict__[k] for k in self.vcl_arguments_order()]
                
    
    def __init__(self, parameters, kernel_prefix):
        self.parameters = parameters
        self.kernel_prefix = kernel_prefix
    
    def make_vcl_template(self):
        return
        
    def check(self, statement):
        vcl_template = self.make_vcl_template(self.parameters.make_parameters());
        vcl_statement = statement.vcl_statement;
        vcl_context = statement.result.context.vcl_sub_context;
        return vcl_statement.check_template(vcl_template, vcl_context);
        
    def execute(self, statement, force_compilation=False):
        vcl_template = self.make_vcl_template(self.parameters.make_parameters());
        vcl_statement = statement.vcl_statement;
        vcl_context = statement.result.context.vcl_sub_context;
        vcl_statement.execute_template(vcl_template, vcl_context, force_compilation);
        return statement.result;


class VectorAxpyTemplate(TemplateBase):
    
    class Parameters(TemplateBase.Parameters):
        
        def __init__(self, simd_width, local_size_0, num_groups_0, decomposition):
            super(VectorAxpyTemplate.Parameters, self).__init__(simd_width, (local_size_0,));
            self.num_groups_0 = num_groups_0;
            self.decomposition = decomposition;
    
        def vcl_arguments_order(self):
            return ['simd_width',  'local_size_0', 'num_groups_0', 'decomposition'];
        
    def __init__(self, parameters, kernel_prefix):
        super(VectorAxpyTemplate, self).__init__(parameters, kernel_prefix);

    def make_vcl_template(self, params):
        return _v.vector_axpy_template(_v.vector_axpy_template.parameters(*params), self.kernel_prefix);

class MatrixAxpyTemplate(TemplateBase):
    
    class Parameters(TemplateBase.Parameters):
        
        def __init__(self, simd_width, local_size_0, local_size_1, num_groups_0, num_groups_1, decomposition):
            super(MatrixAxpyTemplate.Parameters, self).__init__(simd_width, (local_size_0, local_size_1));
            self.num_groups_0 = num_groups_0;
            self.num_groups_1 = num_groups_1;
            self.decomposition = decomposition;
    
        def vcl_arguments_order(self):
            return ['simd_width',  'local_size_0', 'local_size_1', 'num_groups_0', 'num_groups_1', 'decomposition']; 
        
    def __init__(self, parameters, kernel_prefix):
        super(MatrixAxpyTemplate, self).__init__(parameters, kernel_prefix);

    def make_vcl_template(self, params):
        return _v.matrix_axpy_template(_v.matrix_axpy_template.parameters(*params), self.kernel_prefix);
        
        
class ReductionTemplate(TemplateBase):
    
    class Parameters(TemplateBase.Parameters):
        
        def __init__(self, simd_width, local_size_0, num_groups, decomposition):
            super(ReductionTemplate.Parameters, self).__init__(simd_width, (local_size_0,));
            self.num_groups_0 = num_groups_0;
            self.decomposition = decomposition;
    
        def vcl_arguments_order():
            return ['simd_width',  'local_size_0', 'num_groups_0', 'decomposition'];
        
    def __init__(self, parameters, kernel_prefix):
        super(ReductionTemplate, self).__init__(parameters, kernel_prefix);

    def make_vcl_template(self, params):
        return _v.reduction_template(_v.reduction_template.parameters(*params), self.kernel_prefix);
                                
class RowWiseReductionTemplate(TemplateBase):
    
    class Parameters(TemplateBase.Parameters):
        
        def __init__(self,  simd_width, local_size_0, local_size_1, num_groups_0):
            super(RowWiseReductionTemplate.Parameters, self).__init__(simd_width, (local_size_0, local_size_1));
            self.num_groups_0 = num_groups_0;
    
        def vcl_arguments_order(self):
            return ['simd_width',  'local_size_0', 'local_size_1', 'num_groups_0'];
        
    def __init__(self, parameters, A_trans, kernel_prefix):
        super(RowWiseReductionTemplate, self).__init__(parameters, kernel_prefix);
        self.A_trans = A_trans

    def make_vcl_template(self, params):
        return _v.row_wise_reduction_template(_v.row_wise_reduction_template.parameters(*params), self.A_trans, self.kernel_prefix);
        
                                        
class MatrixProductTemplate(TemplateBase):
    
    class Parameters(TemplateBase.Parameters):
        
        def __init__(self, simd_width, local_size_0, kL, local_size_1, mS, kS, nS, use_A_local, use_B_local, local_fetch_0, local_fetch_1):
            super(MatrixProductTemplate.Parameters, self).__init__(simd_width, (local_size_0, local_size_1));
            self.kL = kL;
            self.mS = mS;
            self.kS = kS;
            self.nS = nS;
            self.use_A_local = use_A_local;
            self.use_B_local = use_B_local;
            self.local_fetch_0 = local_fetch_0;
            self.local_fetch_1 = local_fetch_1;
    
        def vcl_arguments_order(self):
            return ['simd_width',  'local_size_0', 'kL', 'local_size_1', 'mS', 'kS', 'nS',  'use_A_local', 'use_B_local', 'local_fetch_0', 'local_fetch_1'];
        
    def __init__(self, parameters, A_trans, B_trans, kernel_prefix):
        super(MatrixProductTemplate, self).__init__(parameters, kernel_prefix);
        self.A_trans = A_trans;
        self.B_trans = B_trans;

    def make_vcl_template(self, params):
        return _v.matrix_product_template(_v.matrix_product_template.parameters(*params), self.A_trans, self.B_trans, self.kernel_prefix);
