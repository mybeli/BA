from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.training.optimizer import Optimizer
import tensorflow as tf

import input_data
import test

class SVRG(Optimizer):
    
    #Values for gate_gradients
    GATE_NONE = 0
    GATE_OP = 1
    GATE_GRAPH = 2
    
    def __init__(self, learning_rate, iter_per_epoch, use_locking=False, name='SVRG'):
        '''
        constructs a new SVRGB optimizer
        '''
        super(SVRG, self).__init__(use_locking, name)
        
        self.learning_rate = learning_rate
        self.iter_per_epoch = iter_per_epoch


    def _create_slots(self, var_list):
        
        for v in var_list:
            with ops.colocate_with(v):
                var_temp = constant_op.constant(0.0,shape=v.get_shape(),dtype=v.dtype.base_dtype)
                
            self._get_or_make_slot(v, var_temp, "var_temp", self._name)
                                    
    def compute_gradient(self, var):
        
        mnist = input_data.read_data_sets('data', one_hot=True, validation_size=0)
      
        x = tf.placeholder(tf.float32, [None, 784])
    
        y = tf.placeholder(tf.float32, [None, 10])
    
        W_fc1 = test.weight_variable([28*28, 1000])
        b_fc1 = test.bias_variable([1000])
        h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
    
        W_fc2 = test.weight_variable([1000, 1000])
        b_fc2 = test.bias_variable([1000])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
      
        W_fc3 = test.weight_variable([1000, 10])
        b_fc3 = test.bias_variable([10])
        out = tf.matmul(h_fc2, W_fc3) + b_fc3
      
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
        
        grad_w1, grad_w2, grad_w3, bias_1, bias_2, bias_3 = tf.gradients(xs = [W_fc1, W_fc2, W_fc3, b_fc1, b_fc2, b_fc3] , ys = cross_entropy)
        
        init = tf.global_variables_initializer
        
        with tf.Session as sess:
            sess.run(init)
            input_x, output_y = mnist.train.next_batch(1)
            grad = sess.run([grad_w1, grad_w2, grad_w3, bias_1, bias_2, bias_3,cross_entropy], feed_dict={x: input_x, y: output_y})
            
       
        return grad.eval(var)
                
    def _apply_dense(self, grad, var):
        
        var_temp = self.get_slot(var, "var_temp")
        
        var_temp = var
        
        for i in range(self.iter_per_epoch):
            grad2 = self.compute_gradient(var_temp)
            grad1 = self.compute_gradient(var)
            var_temp = grad2 - grad1 + grad 
        
        return (var_temp)

    def _apply_sparse(self, grad, var):
        return self._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, handle):
        return self._apply_dense(grad, handle)
    
