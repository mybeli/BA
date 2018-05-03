from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import Optimizer
import tensorflow as tf


class SVRG(Optimizer):
    def __init__(self, alpha=0.1, use_locking=False, name='SVRG'):
        '''
        constructs a new SVRGB optimizer
        '''
        super(SVRG, self).__init__(use_locking, name)
        self._alpha = alpha

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                tilde_delta = constant_op.constant(0, 
                                                     shape=v.get_shape(),
                                                     dtype=v.dtype.base_dtype)
                x = constant_op.constant(0, 
                                                     shape=v.get_shape(),
                                                     dtype=v.dtype.base_dtype)

            self._get_or_make_slot(v, tilde_delta, "tilde_delta", self._name)
            self._get_or_make_slot(v, x, "x", self._name)
          

    # NEW FUNCTION: grad_batch(): calculate gradient of loss function with batch_size = 1
    # return x_k and x_o (weights) in one itration to calculate tilde_delta
    def grad_batch():
        
        return [0,0]

    def _apply_dense(self, grad, var):
        tilde_delta = self.get_slot(var, "tilde_delta")
        x = self.get_slot(var, "x")

        grad_1k, grad_0k = self.grad_batch()

        tilde_delta_update = grad_1k - grad_0k + grad
        x_update = x - self.alpha*tilde_delta_update

        
        tilde_delta_update_op = state_ops.assign(tilde_delta, tilde_delta_update)
        x_update_op = state_ops.assign(x, x_update)

        return control_flow_ops.group(*[tilde_delta_update_op,
                             x_update_op])

    def _apply_sparse(self, grad,grad_1k, grad_0k, var):
        return self._apply_dense(grad,grad_1k, grad_0k, var)

    def _resource_apply_dense(self, grad,grad_1k, grad_0k, handle):
        return self._apply_dense(grad,grad_1k, grad_0k, handle)

