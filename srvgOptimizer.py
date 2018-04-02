#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 23:34:36 2018

@author: mybeli161
"""

from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import Optimizer
import tensorflow as tf


class SVRG(Optimizer):
    def __init__(self, learning_rate, use_locking=False, name='SVRG'):
        '''
        constructs a new SVRG optimizer
        '''
        super(SVRG, self).__init__(use_locking, name)
        self._alpha = learning_rate

    def _create_slots(self, var_list):
        for v in var_list:
            with ops.colocate_with(v):
                x = constant_op.constant(0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)
                tilde_napla = constant_op.constant(0.0, shape=v.get_shape(), dtype=v.dtype.base_dtype)

            self._get_or_make_slot(v, x, "x", self._name)
            self._get_or_make_slot(v, tilde_napla, "tilde_napla", self._name)

    def _apply_dense(self, grad, var):
        tilde_napla = self.get_slot(var, "tilde_napla")
        x = self.get_slot(var, "x")
    
        tilde_napla_update =  tf.+ grad
        x_update = x - self.learning_rate*tilde_napla_update
        
        tilde_napla_update_op = state_ops.assign(tilde_napla,tilde_napla_update)
        x_update_op = state_ops.assign(x, x_update)

        return control_flow_ops.group(*[])
    
    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)
