"""Simple model to regress 3d human poses from 2d joint locations"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import variable_scope as vs

import os
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import data_utils

def kaiming(shape, dtype, partition_info=None):
  """Kaiming initialization as described in https://arxiv.org/pdf/1502.01852.pdf

  Args
    shape: dimensions of the tf array to initialize
    dtype: data type of the array
    partition_info: (Optional) info about how the variable is partitioned.
      See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py#L26
      Needed to be used as an initializer.
  Returns
    Tensorflow array with initial weights
  """
  return(tf.compat.v1.truncated_normal(shape, dtype=dtype)*tf.sqrt(2/float(shape[0])))

class LinearModel(object):
    """ A simple Linear+RELU model """

    def __init__(self,
                 linear_size,
                 num_layers,
                 residual,
                 batch_norm,
                 max_norm,
                 batch_size,
                 learning_rate,
                 summaries_dir,
                 dtype=tf.float32):
        """Creates the linear + relu model

        Args
          linear_size: integer. number of units in each layer of the model
          num_layers: integer. number of bilinear blocks in the model
          residual: boolean. Whether to add residual connections
          batch_norm: boolean. Whether to use batch normalization
          max_norm: boolean. Whether to clip weights to a norm of 1
          batch_size: integer. The size of the batches used during training
          learning_rate: float. Learning rate to start with
          summaries_dir: String. Directory where to log progress
          dtype: the data type to use to store internal variables
        """

        # TODO: check these two values
        self.FLY_2D_SIZE = len(data_utils.LEGS_2D)-len(data_utils.JOINTS_IGNORE)
        self.FLY_3D_SIZE = len(data_utils.LEGS)-len(data_utils.JOINTS_IGNORE)

        self.input_size  = self.FLY_2D_SIZE
        self.output_size = self.FLY_3D_SIZE

        self.isTraining = tf.compat.v1.placeholder(tf.bool,name="isTrainingflag")
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")

        # Summary writers for train and test runs
        self.train_writer = tf.compat.v1.summary.FileWriter( os.path.join(summaries_dir, 'train' ))
        self.test_writer  = tf.compat.v1.summary.FileWriter( os.path.join(summaries_dir, 'test' ))

        self.linear_size   = linear_size
        self.batch_size    = batch_size
        self.learning_rate = tf.Variable( float(learning_rate), trainable=False,
                                          dtype=dtype, name="learning_rate")
        self.global_step   = tf.Variable(0, trainable=False, name="global_step")
        decay_steps = 100000  # empirical
        decay_rate = 0.96     # empirical
        self.learning_rate =  tf.compat.v1.train.exponential_decay(
          self.learning_rate, self.global_step, decay_steps, decay_rate)

        # === Transform the inputs ===
        with vs.variable_scope("inputs"):

            # in=2d poses, out=3d poses
            enc_in  = tf.compat.v1.placeholder(dtype, shape=[None, self.input_size], name="enc_in")
            dec_out = tf.compat.v1.placeholder(dtype, shape=[None, self.output_size], name="dec_out")

            self.encoder_inputs  = enc_in
            self.decoder_outputs = dec_out

        # === Create the linear + relu combos ===
        with vs.variable_scope( "linear_model" ):

            # === First layer, brings dimensionality up to linear_size ===
            w1 = tf.compat.v1.get_variable(name="w1", initializer=kaiming,
                                           shape=[self.FLY_2D_SIZE, linear_size], dtype=dtype )
            b1 = tf.compat.v1.get_variable(name="b1", initializer=kaiming,
                                           shape=[linear_size], dtype=dtype )
            w1 = tf.clip_by_norm(w1,1) if max_norm else w1
            y3 = tf.matmul( enc_in, w1 ) + b1

            if batch_norm:
                y3 = tf.layers.batch_normalization(
                    y3, training=self.isTraining, name="batch_normalization")
            y3 = tf.nn.relu( y3 )
            y3 = tf.nn.dropout( y3, rate = 1-self.dropout_keep_prob )

            # === Create multiple bi-linear layers ===
            for idx in range( num_layers ):
                y3 = self.two_linear( y3, linear_size, residual, self.dropout_keep_prob,
                                      max_norm, batch_norm, dtype, idx )

            # === Last linear layer has HUMAN_3D_SIZE in output ===
            w4 = tf.get_variable( name="w4", initializer=kaiming,
                                  shape=[linear_size, self.FLY_3D_SIZE], dtype=dtype )
            b4 = tf.get_variable( name="b4", initializer=kaiming, shape=[self.FLY_3D_SIZE], dtype=dtype )
            w4 = tf.clip_by_norm(w4,1) if max_norm else w4
            y = tf.matmul(y3, w4) + b4
            # === End linear model ===

        # Store the outputs here
        self.outputs = y
        self.loss = tf.reduce_mean(tf.square(y - dec_out))
        self.loss_summary = tf.compat.v1.summary.scalar('loss/loss', self.loss)

        # To keep track of the loss in mm
        self.err_mm = tf.placeholder( tf.float32, name="error_mm" )
        self.err_mm_summary = tf.summary.scalar( "loss/error_mm", self.err_mm )

        # Gradients and update operation for training the model.
        opt = tf.compat.v1.train.AdamOptimizer( self.learning_rate )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):

            # Update all the trainable parameters
            gradients = opt.compute_gradients(self.loss)
            self.gradients = [[] if i==None else i for i in gradients]
            self.updates = opt.apply_gradients(gradients, global_step=self.global_step)

        # Keep track of the learning rate
        self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

        # To save the model
        self.saver = tf.compat.v1.train.Saver( tf.global_variables(), max_to_keep=10 )


    def two_linear(self, xin, linear_size, residual, dropout_keep_prob,
                   max_norm, batch_norm, dtype, idx):
        """
        Make a bi-linear block with optional residual connection

        Args
          xin: the batch that enters the block
          linear_size: integer. The size of the linear units
          residual: boolean. Whether to add a residual connection
          dropout_keep_prob: float [0,1]. Probability of dropping something out
          max_norm: boolean. Whether to clip weights to 1-norm
          batch_norm: boolean. Whether to do batch normalization
          dtype: type of the weigths. Usually tf.float32
          idx: integer. Number of layer (for naming/scoping)
        Returns
          y: the batch after it leaves the block
        """

        with vs.variable_scope( "two_linear_"+str(idx) ) as scope:

            input_size = int(xin.get_shape()[1])

            # Linear 1
            w2 = tf.get_variable(name="w2_"+str(idx), initializer=kaiming,
                                 shape=[input_size, linear_size], dtype=dtype)
            b2 = tf.get_variable(name="b2_"+str(idx), initializer=kaiming,
                                 shape=[linear_size], dtype=dtype)
            w2 = tf.clip_by_norm(w2,1) if max_norm else w2
            y = tf.matmul(xin, w2) + b2
            if  batch_norm:
                y = tf.layers.batch_normalization(y, training=self.isTraining,
                                              name="batch_normalization1"+str(idx))

            y = tf.nn.relu( y )
            y = tf.nn.dropout( y, dropout_keep_prob )

            # Linear 2
            w3 = tf.get_variable(name="w3_"+str(idx), initializer=kaiming,
                                 shape=[linear_size, linear_size], dtype=dtype)
            b3 = tf.get_variable(name="b3_"+str(idx), initializer=kaiming, 
                                 shape=[linear_size], dtype=dtype)
            w3 = tf.clip_by_norm(w3,1) if max_norm else w3
            y = tf.matmul(y, w3) + b3

            if  batch_norm:
                y = tf.layers.batch_normalization(y, training=self.isTraining,
                                              name="batch_normalization2"+str(idx))

            y = tf.nn.relu( y )
            y = tf.nn.dropout( y, dropout_keep_prob )

            # Residual every 2 blocks
            y = (xin + y) if residual else y

        return y

    def step(self, session, encoder_inputs, decoder_outputs, dropout_keep_prob, isTraining=True):
        """Run a step of the model feeding the given inputs.

        Args
          session: tensorflow session to use
          encoder_inputs: list of numpy vectors to feed as encoder inputs
          decoder_outputs: list of numpy vectors that are the expected decoder outputs
          dropout_keep_prob: (0,1] dropout keep probability
          isTraining: whether to do the backward step or only forward

        Returns
          if isTraining is True, a 4-tuple
            loss: the computed loss of this batch
            loss_summary: tf summary of this batch loss, to log on tensorboard
            learning_rate_summary: tf summary of learnign rate to log on tensorboard
            outputs: predicted 3d poses
          if isTraining is False, a 3-tuple
            (loss, loss_summary, outputs) same as above
        """

        input_feed = {self.encoder_inputs: encoder_inputs,
                      self.decoder_outputs: decoder_outputs,
                      self.isTraining: isTraining,
                      self.dropout_keep_prob: dropout_keep_prob}

        # Output feed: depends on whether we do a backward step or not.
        if isTraining:
            output_feed = [self.updates,       # Update Op that does SGD
                           self.loss,
                           self.loss_summary,
                           self.learning_rate_summary,
                           self.outputs]

            outputs = session.run( output_feed, input_feed )
            return outputs[1], outputs[2], outputs[3], outputs[4]

        else:
            output_feed = [self.loss, # Loss for this batch.
                           self.loss_summary,
                           self.outputs]

            outputs = session.run(output_feed, input_feed)
            return outputs[0], outputs[1], outputs[2]  # No gradient norm

    def get_all_batches( self, data_x, data_y, training=True ):
        """
        Obtain a list of all the batches, randomly permutted
        Args
          data_x: dictionary with 2d inputs
          data_y: dictionary with 3d expected outputs
          training: True if this is a training batch. False otherwise.
        Returns
          encoder_inputs: list of 2d batches
          decoder_outputs: list of 3d batches
        """

        # Figure out how many frames we have
        n = data_x.shape[0] 
        
        encoder_inputs  = np.reshape(data_x, (-1, 2*data_x.shape[1]))
        decoder_outputs = np.copy(data_y)

        if training:
            # Randomly permute everything
            idx = np.random.permutation( n )
            encoder_inputs  = encoder_inputs[idx, :]
            decoder_outputs = decoder_outputs[idx, :]

        # Make the number of examples a multiple of the batch size
        n_extra  = n % self.batch_size
        if n_extra > 0:  # Otherwise examples are already a multiple of batch size
            encoder_inputs  = encoder_inputs[:-n_extra, :]
            decoder_outputs = decoder_outputs[:-n_extra, :]
        
        n_batches = n // self.batch_size
        encoder_inputs  = np.split(encoder_inputs, n_batches)
        decoder_outputs = np.split(decoder_outputs, n_batches)

        return encoder_inputs, decoder_outputs
