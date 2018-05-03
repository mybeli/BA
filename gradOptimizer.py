from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import Optimizer
import tensorflow as tf


class GradOpt(Optimizer):
    def __init__(self, learning_rate, use_locking=False, name='GradOpt'):
        '''
        constructs a new GradOpt optimizer
        '''
        super(GradOpt, self).__init__(use_locking, name)
        self._alpha = learning_rate

    def _create_slots(self, var_list):


    def _apply_dense(self, grad, var):    
        #HERE TODO

        return control_flow_ops.group(*[])
    
    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)
