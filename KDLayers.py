import tensorflow as tf
from tensorflow.keras.layers import Layer

class KDRightLayer(Layer):
  def __init__(self,n,activation=None):
    super(KDRightLayer, self).__init__()
    self.n = n
    self.activation = tf.keras.activations.get(activation)

  def build(self, input_shape):
    self.q = input_shape[1]
    self.m = input_shape[2]
    # initialize the weights
    self.w = self.add_weight("kernel", shape=[self.m, self.n],trainable=True)
    # initialize the biases
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(name="bias",initial_value=b_init(shape=(self.q,self.n), dtype='float32'),
                         trainable=True)

  def call(self, inputs):
    return self.activation(tf.matmul(inputs, self.w) + self.b)

class KDLeftLayer(Layer):
  def __init__(self,m,activation=None):
    super(KDLeftLayer, self).__init__()
    self.m = m
    self.activation = tf.keras.activations.get(activation)

  def build(self, input_shape):
    self.n = input_shape[1]
    self.p = input_shape[2]
    # initialize the weights
    self.w = self.add_weight("kernel", shape=[self.m, self.n],trainable=True)
    # initialize the biases
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(name="bias",initial_value=b_init(shape=(self.m,self.p), 
                                                          dtype='float32'),trainable=True)

  def call(self, inputs):
    return self.activation(tf.transpose(tf.matmul(tf.transpose(inputs,perm=[0, 2, 1]),
                                                  tf.transpose(self.w)),perm=[0, 2, 1]) + self.b)
