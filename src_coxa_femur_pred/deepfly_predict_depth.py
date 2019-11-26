
"""Predicting 3d poses from 2d joints"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import random
import time
import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import linear_model

tf.app.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate")
tf.app.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability. 1 means no dropout")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size to use during training")
tf.app.flags.DEFINE_integer("epochs", 50, "How many epochs we should train for")
tf.app.flags.DEFINE_boolean("max_norm", True, "Apply maxnorm constraint to the weights")
tf.app.flags.DEFINE_boolean("batch_norm", True, "Use batch_normalization")

# Architecture
tf.app.flags.DEFINE_integer("linear_size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_boolean("residual", True, "Whether to add a residual connection every 2 layers")

# Directories
tf.app.flags.DEFINE_string("train_dir", "experiments_depth/test", "Training directory.")

# Train or load
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("load", 0, "Try to load a previous checkpoint.")

# Misc
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")

FLAGS = tf.app.flags.FLAGS

train_dir = FLAGS.train_dir
train_dir += "_"+str(FLAGS.epochs)+"epochs"
train_dir += "_size"+str(FLAGS.linear_size)
train_dir += "_dropout"+str(FLAGS.dropout)
train_dir += "_"+str(FLAGS.learning_rate)

print("\n\n[*] training directory: ", train_dir)
summaries_dir = os.path.join(train_dir, "log") # Directory for TB summaries

# To avoid race conditions: https://github.com/tensorflow/tensorflow/issues/7448
os.system('mkdir -p {}'.format(summaries_dir))

def create_model( session, batch_size ):
  """
  Create model and initialize it or load its parameters in a session
  Args
    session: tensorflow session
    batch_size: integer. Number of examples in each batch
  Returns
    model: The created (or loaded) model
  Raises
    ValueError if asked to load a model, but the checkpoint specified by
    FLAGS.load cannot be found.
  """
  model = linear_model.LinearModel(
      FLAGS.linear_size,
      FLAGS.num_layers,
      FLAGS.residual,
      FLAGS.batch_norm,
      FLAGS.max_norm,
      batch_size,
      FLAGS.learning_rate,
      summaries_dir,
      dtype=tf.float16 if FLAGS.use_fp16 else tf.float32)

  if FLAGS.load <= 0:
    # Create a new model from scratch
    print("Creating model with fresh parameters.")
    session.run( tf.global_variables_initializer() )
    return model

  # Load a previously saved model
  ckpt = tf.train.get_checkpoint_state( train_dir, latest_filename="checkpoint")
  print( "train_dir", train_dir )

  if ckpt and ckpt.model_checkpoint_path:
    # Check if the specific cpixels = pixels / pixels[2,:]heckpoint exists
    if FLAGS.load > 0:
      if os.path.isfile(os.path.join(train_dir,"checkpoint-{0}.index".format(FLAGS.load))):
        ckpt_name = os.path.join( os.path.join(train_dir,"checkpoint-{0}".format(FLAGS.load)) )
      else:
        raise ValueError("Asked to load checkpoint {0}, but it does not seem to exist".format(FLAGS.load))
    else:
      ckpt_name = os.path.basename( ckpt.model_checkpoint_path )

    print("Loading model {0}".format( ckpt_name ))
    model.saver.restore( session, ckpt.model_checkpoint_path )
    return model
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint {0} does not seem to exist".format( ckpt.model_checkpoint_path ) )

  return model

def test():
  
  # Load 3d data and 2d projections
  df_data, ob_data_3d, dims, ob_dims, train_test, data_mean_3d, data_std_3d,\
      ob_dtu, ob_dtu_flat_2d, ob_dtu_flat_xy, df_dtu, df_dtu_flat_3d = data_utils.read_3d_data()
  
  unNorm_df_data = data_utils.unNormalize_data(df_data, data_mean_3d, data_std_3d, df_dtu_flat_3d)
  
  print("\n[+] done reading and normalizing data")
  # Getting the number of training and test subjects
  test2d, test_depth = [], []
  test3d_full = []
  for idx, t in enumerate(train_test):
    if t == 0: # test data
      d3d = df_data[dims[idx]:dims[idx+1]]
      test3d_full.append(unNorm_df_data[dims[idx]:dims[idx+1]])

      td = d3d[:,:, data_utils.Z_COORD]
      cf2d = d3d[:, data_utils.DF_COXA_FEMUR]
      cf2d = cf2d[:,:, data_utils.XY_COORD].reshape((cf2d.shape[0], -1))
      test_depth.append(np.hstack([td[:, df_dtu], cf2d]))

      t2d = d3d[:, data_utils.DF_NON_COXA_FEMUR]
      t2d = t2d[:,:, data_utils.XY_COORD]
      test2d.append(np.delete(t2d, data_utils.JOINTS_IGNORE, axis=1))

  test2d = np.vstack(test2d)
  test_depth = np.vstack(test_depth)
  test3d_full = np.vstack(test3d_full)
  n_extra  = test3d_full.shape[0] % FLAGS.batch_size
  if n_extra > 0:  # Otherwise examples are already a multiple of batch size
    test3d_full = test3d_full[:-n_extra]

  print(df_dtu)
  print("3D data mean:")
  print(data_mean_3d)
  print("3D data std:")
  print(data_std_3d)
  
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    device_count=device_count)) as sess:
    
    # === Create the model ===
    print("[*] creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    batch_size = FLAGS.batch_size
    # Dropout probability 0 (keep probability 1) for sampling
    dp = 1.0
  
    n_joints = len( df_dtu ) + len(data_utils.DF_COXA_FEMUR)*2

    model = create_model(sess, batch_size)
    print("[+] model loaded")
    # faking a decoder output, even if optobot data DO NOT have depth fround truth
    encoder_inputs, decoder_outputs =\
            model.get_all_batches(test2d, test_depth, training=False)
  
    nbatches = len(encoder_inputs)
    print("[*] there are {0} test batches".format(nbatches))

    # predicted depths
    all_depth_predic = []
    all_dists = []
    for i in range( nbatches ):
      enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
      _, _, depth_predic = model.step(sess, enc_in, dec_out, dp, isTraining=False)
      
      # Un-normalize
      data_mean_3d = np.reshape(data_mean_3d, (data_utils.DF_NUM_JOINTS, 3))
      data_std_3d = np.reshape(data_std_3d, (data_utils.DF_NUM_JOINTS, 3))

      dm1d = data_mean_3d[:, data_utils.Z_COORD] # keeping only the depth
      # adding the COXA FEMUR x-y coordinate to the end
      dmcf2d = data_mean_3d[data_utils.DF_COXA_FEMUR]
      dmcf2d = dmcf2d[:, data_utils.XY_COORD].flatten()
      dm1d2d = np.concatenate((dm1d, dmcf2d), axis=0)

      ds1d = data_std_3d[:, data_utils.Z_COORD] # keeping only the depth
      # adding the COXA FEMUR x-y coordinate to the end
      dscf2d = data_mean_3d[data_utils.DF_COXA_FEMUR]
      dscf2d = dscf2d[:, data_utils.XY_COORD].flatten()
      ds1d2d = np.concatenate((ds1d, dscf2d), axis=0)

      df_dtu_with_coxa = df_dtu +\
          list(range(max(df_dtu)+1, max(df_dtu)+1+len(data_utils.DF_COXA_FEMUR)*2))

      dec_out = data_utils.unNormalize_batch(dec_out, dm1d2d, ds1d2d, df_dtu_with_coxa)
      depth_predic = data_utils.unNormalize_batch(depth_predic, dm1d2d, ds1d2d, df_dtu_with_coxa)

      dec_out_red = dec_out[:, df_dtu_with_coxa]
      depth_predic_red = depth_predic[:, df_dtu_with_coxa]

      assert dec_out.shape[0] == FLAGS.batch_size
      assert depth_predic.shape[0] == FLAGS.batch_size

      all_dists = np.abs(depth_predic_red - dec_out_red)
      
      all_depth_predic.append(depth_predic)

    # Error per joint, per coordinate and total for all passed batches
    joint_err = np.mean( all_dists, axis=0 )
    total_err = np.mean( all_dists )
    print_ = "=============================\n" \
      "Val error avg (mm):  %.2f\n" \
      "=============================\n" % ( total_err )

    for i in range(n_joints):
      print_ += "Error in joint {0:02d} (mm): {1:>5.2f}\n".format(i+1, joint_err[i])
    print_ += "============================="
    with open(train_dir+"/info.txt", "w+") as f:
      f.write(print_)
    print(print_)

    # Put all the predicted depths together
    depth_predic = np.vstack(all_depth_predic)
    print(depth_predic.shape)

    df_predic = np.zeros(test3d_full.shape)
    df_predic[:,:,data_utils.XY_COORD] = test3d_full[:,:,data_utils.XY_COORD]
    inc = 1
    for cf in data_utils.DF_COXA_FEMUR:
      for coord in data_utils.XY_COORD:
        df_predic[:, cf, coord] = depth_predic[:, len(df_dtu)+inc]
        inc += 1
    
    df_predic[:,:,data_utils.Z_COORD] = depth_predic[:, :len(df_dtu)+1]
    
    np.save("saved_structures/df_test_data.npy", test3d_full)
    np.save("saved_structures/df_predic.npy", df_predic)

def main(_):
    test()

if __name__ == "__main__":
  tf.compat.v1.app.run()
