
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
tf.app.flags.DEFINE_boolean("test", False, "Set to True for testing.")
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

def train():
  """Train a linear model for 3d pose estimation"""

  # Load 3d data and 2d projections
  df_data, ob_data_3d, dims, ob_dims, train_test, data_mean_3d,\
  data_std_3d, ob_dtu, ob_dtu_flat_2d, ob_dtu_flat_xy, df_dtu, df_dtu_flat_3d =\
    data_utils.read_3d_data()
  
  print("\n[+] done reading and normalizing data")
  # Getting the number of training and test subjects
  tr_subj, te_subj = 0, 0
  train_depth, train2d, test_depth, test2d = [], [], [], []
  for idx, t in enumerate(train_test):
    if t == 1: # train data
      d3d = df_data[dims[idx]:dims[idx+1]]
      td = d3d[:,:, data_utils.Z_COORD]
      cf2d = d3d[:, data_utils.DF_COXA_FEMUR]
      cf2d = cf2d[:,:, data_utils.XY_COORD].reshape((cf2d.shape[0], -1))
      train_depth.append(np.hstack([td[:, df_dtu], cf2d]))
      
      t2d = d3d[:, data_utils.DF_NON_COXA_FEMUR]
      t2d = t2d[:,:, data_utils.XY_COORD]
      train2d.append(np.delete(t2d, data_utils.JOINTS_IGNORE, axis=1))
      tr_subj += dims[idx+1]-dims[idx]
    else: # test data
      d3d = df_data[dims[idx]:dims[idx+1]]
      td = d3d[:,:, data_utils.Z_COORD]
      cf2d = d3d[:, data_utils.DF_COXA_FEMUR]
      cf2d = cf2d[:,:, data_utils.XY_COORD].reshape((cf2d.shape[0], -1))
      test_depth.append(np.hstack([td[:, df_dtu], cf2d]))
      
      t2d = d3d[:, data_utils.DF_NON_COXA_FEMUR]
      t2d = t2d[:,:, data_utils.XY_COORD]
      test2d.append(np.delete(t2d, data_utils.JOINTS_IGNORE, axis=1))
      te_subj += dims[idx+1]-dims[idx]

  train_depth = np.vstack(train_depth)
  train2d = np.vstack(train2d)
  test_depth = np.vstack(test_depth)
  test2d = np.vstack(test2d)
  print(train2d.shape, train_depth.shape)
  print(test2d.shape, test_depth.shape)
  print("{0} training subjects, {1} test subjects".format(tr_subj, te_subj))
  print(df_dtu)

  print("3D data mean:")
  print(data_mean_3d)
  print("3D data std:")
  print(data_std_3d)
  
  input("Press Enter to continue...")

  # Avoid using the GPU if requested
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}
  with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
    device_count=device_count, allow_soft_placement=True )) as sess:

    # === Create the model ===
    print("[*] creating %d bi-layers of %d units." % (FLAGS.num_layers, FLAGS.linear_size))
    model = create_model( sess, FLAGS.batch_size )
    model.train_writer.add_graph( sess.graph )
    print("[+] model created")
    
    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
    previous_losses = []

    step_time, loss = 0, 0
    current_epoch = 0
    log_every_n_batches = 100
    losses, errors, joint_errors = [], [], []
    for _ in range( FLAGS.epochs ):
      current_epoch = current_epoch + 1

      # === Load training batches for one epoch ===
      encoder_inputs, decoder_outputs =\
         model.get_all_batches( train2d, train_depth, training=True )
      nbatches = len( encoder_inputs )
      print("[*] there are {0} train batches".format( nbatches ))
      start_time, loss = time.time(), 0.
      # === Loop through all the training batches ===
      for i in range( nbatches ):

        if (i+1) % log_every_n_batches == 0:
          # Print progress every log_every_n_batches batches
          print("Working on epoch {0}, batch {1} / {2}...".format( current_epoch, i+1, nbatches),end="" )

        enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
        step_loss, loss_summary, lr_summary, _ =\
          model.step( sess, enc_in, dec_out, FLAGS.dropout, isTraining=True )

        if (i+1) % log_every_n_batches == 0:
          # Log and print progress every log_every_n_batches batchespixels = pixels / pixels[2,:]
          model.train_writer.add_summary( loss_summary, current_step )
          model.train_writer.add_summary( lr_summary, current_step )
          step_time = (time.time() - start_time)
          start_time = time.time()
          print("done in {0:.2f} ms".format( 1000*step_time / log_every_n_batches ) )

        loss += step_loss
        current_step += 1
        # === end looping through training batches ===

      loss = loss / nbatches
      losses.append(loss)
      print("=============================\n"
            "Global step:         %d\n"
            "Learning rate:       %.2e\n"
            "Train loss avg:      %.4f\n"
            "=============================" % (model.global_step.eval(),
            model.learning_rate.eval(), loss) )
      # === End training for an epoch ===

      # === Testing after this epoch ===
      isTraining = False
      
      n_joints = len(df_dtu) + len(data_utils.DF_COXA_FEMUR)*2

      encoder_inputs, decoder_outputs =\
         model.get_all_batches( test2d, test_depth, training=False)

      total_err, joint_err, step_time, loss = evaluate_batches( sess, model,
        data_mean_3d, data_std_3d, df_dtu, encoder_inputs, decoder_outputs, current_epoch )

      print("=============================\n"
            "Step-time (ms):      %.4f\n"
            "Val loss avg:        %.4f\n"
            "Val error avg (mm):  %.2f\n"
            "=============================" % ( 1000*step_time, loss, total_err))
      
      for i in range(n_joints):
        # 6 spaces, right-aligned, 5 decimal places
        print("Error in joint {0:02d} (mm): {1:>5.2f}".format(i+1, joint_err[i]))
      print("=============================")
      errors.append(total_err)
      joint_errors.append(joint_err)
      # Log the error to tensorboard
      summaries = sess.run( model.err_mm_summary, {model.err_mm: total_err} )
      model.test_writer.add_summary( summaries, current_step )

      # Save the model
      print( "Saving the model... ", end="" )
      start_time = time.time()
      model.saver.save(sess, os.path.join(train_dir, 'checkpoint'), global_step=current_step )
      print( "done in {0:.2f} ms".format(1000*(time.time() - start_time)) )

      # Reset global time and loss
      step_time, loss = 0, 0

      sys.stdout.flush()
    # Save losses for future plots
    def print_list_tofile(l, filename):
      with open(filename, 'wb') as f:
        pickle.dump(l, f)
    print_list_tofile(losses, train_dir+"/losses.pkl")
    print_list_tofile(errors, train_dir+"/errors.pkl")
    print_list_tofile(joint_errors, train_dir+"/joint_errors.pkl")

def evaluate_batches( sess, model, data_mean_3d, data_std_3d, df_dtu,
  encoder_inputs, decoder_outputs, current_epoch ):
  """
  Generic method that evaluates performance of a list of batches.
  May be used to evaluate all actions or a single action.
  """

  nbatches = len( encoder_inputs )

  # Loop through test examples
  all_dists, start_time, loss = [], time.time(), 0.
  log_every_n_batches = 20
  for i in range(nbatches):

    if current_epoch > 0 and (i+1) % log_every_n_batches == 0:
      print("Working on test epoch {0}, batch {1} / {2}".format( current_epoch, i+1, nbatches) )

    enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
    dp = 1.0 # dropout keep probability is always 1 at test time
    step_loss, loss_summary, poses3d = model.step( sess, enc_in, dec_out, dp, isTraining=False )
    loss += step_loss

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
    poses3d = data_utils.unNormalize_batch(poses3d, dm1d2d, ds1d2d, df_dtu_with_coxa)

    # Keep only the relevant dimensions
    dec_out = dec_out[:, df_dtu_with_coxa]
    poses3d = poses3d[:, df_dtu_with_coxa]

    assert dec_out.shape[0] == FLAGS.batch_size
    assert poses3d.shape[0] == FLAGS.batch_size
    
    # Compute Euclidean distance error per joint
    all_dists = np.abs(poses3d - dec_out) # Array with L2 error per joint in mm

  step_time = (time.time() - start_time) / nbatches
  loss      = loss / nbatches

  # Error per joint and total for all passed batches
  joint_err = np.mean( all_dists, axis=0 )
  total_err = np.mean( all_dists )
  
  return total_err, joint_err, step_time, loss

def test():
  # Load 3d data and 2d projections
  df_data, ob_data_3d, dims, ob_dims, train_test, data_mean_3d,\
  data_std_3d, ob_dtu, ob_dtu_flat_2d, ob_dtu_flat_xy, df_dtu, df_dtu_flat_3d =\
    data_utils.read_3d_data()

  unNorm_ob_data_3d = np.copy(ob_data_3d)
  unNorm_ob_data_3d[:,:,data_utils.XY_COORD] =\
      data_utils.unNormalize_ob_data(unNorm_ob_data_3d[:,:,data_utils.XY_COORD],
                                     data_mean_3d[data_utils.DF_NON_COXA_FEMUR_3D],
                                     data_std_3d[data_utils.DF_NON_COXA_FEMUR_3D],
                                     ob_dtu_flat_2d, ob_dtu_flat_xy)

  # keep only X, Y coordinate, because this is the input of the NN
  ob_data = ob_data_3d[:,:,data_utils.XY_COORD]

  print("\n[+] done reading and normalizing data")
  # Getting the number of training and test subjects
  test_depth = []
  for idx, t in enumerate(train_test):
    if t == 0: # test data
      d3d = df_data[dims[idx]:dims[idx+1]]
      td = d3d[:,:, data_utils.Z_COORD]
      cf2d = d3d[:, data_utils.DF_COXA_FEMUR]
      cf2d = cf2d[:,:, data_utils.XY_COORD].reshape((cf2d.shape[0], -1))
      test_depth.append(np.hstack([td[:, df_dtu], cf2d]))

  test_depth = np.vstack(test_depth)
  # keeping only a part of test3d data, in order to fake a decoder output
  # even if optobot data DO NOT have depth ground truth
  test_depth = test_depth[:ob_data_3d.shape[0]]
  n_extra  = unNorm_ob_data_3d.shape[0] % FLAGS.batch_size
  if n_extra > 0:  # Otherwise examples are already a multiple of batch size
    unNorm_ob_data_3d = unNorm_ob_data_3d[:-n_extra]
    ob_data = ob_data[:-n_extra]
    test_depth = test_depth[:-n_extra]
  assert unNorm_ob_data_3d.shape[0] == test_depth.shape[0]

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
    print(ob_data[:,ob_dtu].shape)
    # faking a decoder output, even if optobot data DO NOT have depth ground truth
    ob_encoder_inputs, decoder_outputs =\
            model.get_all_batches(ob_data[:,ob_dtu], test_depth, training=False)
  
    nbatches = len(ob_encoder_inputs)
    print("[*] there are {0} test batches".format(nbatches))

    # predicted depths
    all_ob_predic = []
    for i in range( nbatches ):
      enc_in = ob_encoder_inputs[i]
      # faking a decoder output, even if optobot data DO NOT have depth fround truth
      dec_out = decoder_outputs[i]
      _, _, ob_predic = model.step(sess, enc_in, dec_out, dp, isTraining=False)
      
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

      ob_predic = data_utils.unNormalize_batch(ob_predic, dm1d2d, ds1d2d, df_dtu_with_coxa)

      assert ob_predic.shape[0] == FLAGS.batch_size
      
      all_ob_predic.append(ob_predic)

    # Put all the predicted depths together
    ob_predic = np.vstack(all_ob_predic)
    print(ob_predic.shape)
    print(unNorm_ob_data_3d.shape)
    
    unNorm_ob_data_3d[:,:,data_utils.Z_COORD] = ob_predic[:,data_utils.DF_NON_COXA_FEMUR]
    
    np.save("saved_structures/ob_predic.npy", unNorm_ob_data_3d)

def main(_):
  if FLAGS.test:
    test()
  else:
    train()

if __name__ == "__main__":
  tf.compat.v1.app.run()
