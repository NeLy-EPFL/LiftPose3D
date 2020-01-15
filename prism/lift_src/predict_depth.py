
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
tf.app.flags.DEFINE_string("train_dir", "experiments_lifted/test", "Training directory.")

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
    df_data, pr_data, df_dims, pr_dims, train_test, data_mean, data_std, dtu, dtu_flat =\
        data_utils.read_3d_data()
  
    print("\n[+] done reading and normalizing data")
    # Getting the number of training and test subjects
    tr_subj, te_subj = 0, 0
    train_depth, train2d, test_depth, test2d = [], [], [], []
    for idx, t in enumerate(train_test):
        if t == 1: # train data
            d3d = df_data[dims[idx]:dims[idx+1]]
            td = d3d[:,:, data_utils.DEPTH_COORD]
            train_depth.append(td[:, dtu])
      
            t2d = d3d[:,:, data_utils.XY_COORD]
            train2d.append(t2d[:, dtu])
            tr_subj += dims[idx+1]-dims[idx]
        else: # test data
            d3d = df_data[dims[idx]:dims[idx+1]]
            td = d3d[:,:, data_utils.DEPTH_COORD]
            test_depth.append(td[:, dtu])
          
            t2d = d3d[:,:, data_utils.XY_COORD]
            test2d.append(t2d[:, dtu])
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
        model = create_model(sess, FLAGS.batch_size)
        model.train_writer.add_graph(sess.graph)
        print("[+] model created")
        
        #=== This is the training loop ===
        step_time, loss, val_loss = 0.0, 0.0, 0.0
        current_step = 0 if FLAGS.load <= 0 else FLAGS.load + 1
        previous_losses = []

        step_time, loss = 0, 0
        log_every_n_batches = 100
        losses, errors, joint_errors = [], [], []
        for current_epoch in range(FLAGS.epochs):

            # === Load training batches for one epoch ===
            encoder_inputs, decoder_outputs =\
                 model.get_all_batches(train2d, train_depth, training=True)
            nbatches = len(encoder_inputs)
            print("[*] there are {0} train batches".format( nbatches ))
            start_time, loss = time.time(), 0.
            # === Loop through all the training batches ===
            for i in range(nbatches):

                if (i+1) % log_every_n_batches == 0:
                    # Print progress every log_every_n_batches batches
                    print("Working on epoch {0}, batch {1} / {2}..."\
                        .format(current_epoch, i+1, nbatches), end="")

                enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
                step_loss, loss_summary, lr_summary, _ =\
                    model.step(sess, enc_in, dec_out, FLAGS.dropout, isTraining=True)

                if (i+1) % log_every_n_batches == 0:
                    # Log and print progress every log_every_n_batches
                    # batchespixels = pixels / pixels[2,:]
                    model.train_writer.add_summary(loss_summary, current_step)
                    model.train_writer.add_summary(lr_summary, current_step)
                    step_time = time.time() - start_time
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
          
            n_joints = len(dtu)

            encoder_inputs, decoder_outputs =\
                model.get_all_batches(test2d, test_depth, training=False)

            total_err, joint_err, step_time, loss = evaluate_batches(sess, model,
                data_mean, data_std, dtu, encoder_inputs, decoder_outputs, current_epoch)

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

def evaluate_batches(sess, model, data_mean, data_std, dtu,
    encoder_inputs, decoder_outputs, current_epoch):
    """
    Generic method that evaluates performance of a list of batches.
    May be used to evaluate all actions or a single action.
    """

    nbatches = len(encoder_inputs)

    # Loop through test examples
    all_dists, start_time, loss = [], time.time(), 0.
    log_every_n_batches = 20
    for i in range(nbatches):

        if current_epoch > 0 and (i+1) % log_every_n_batches == 0:
            print("Working on test epoch {0}, batch {1} / {2}".format( current_epoch, i+1, nbatches) )

        enc_in, dec_out = encoder_inputs[i], decoder_outputs[i]
        dp = 1.0 # dropout keep probability is always 1 at test time
        step_loss, loss_summary, poses3d = model.step(sess, enc_in, dec_out, dp, isTraining=False)
        loss += step_loss

        data_mean = np.reshape(data_mean, (data_utils.NUM_JOINTS, 3))
        data_std = np.reshape(data_std, (data_utils.NUM_JOINTS, 3))
        
        data_mean_depth = data_mean[:, data_utils.DEPTH_COORD]
        
        data_std_depth = data_std[:, data_utils.DEPTH_COORD]

        dec_out = data_utils.unNormalize_batch(dec_out, data_mean_depth, data_std_depth, dtu)
        poses3d = data_utils.unNormalize_batch(poses3d, data_mean_depth, data_std_depth, dtu)

        # Keep only the relevant dimensions
        dec_out = dec_out[:, dtu]
        poses3d = poses3d[:, dtu]

        assert dec_out.shape[0] == FLAGS.batch_size
        assert poses3d.shape[0] == FLAGS.batch_size
        
        # Compute Euclidean distance error per joint
        all_dists.append(np.abs(poses3d - dec_out)) # Array with L2 error per joint in mm

    all_dists = np.vstack(all_dists)

    step_time = (time.time() - start_time) / nbatches
    loss      = loss / nbatches

    # Error per joint and total for all passed batches
    joint_err = np.mean(all_dists, axis=0)
    total_err = np.mean(all_dists)
  
    return total_err, joint_err, step_time, loss

def test():
    # Load 3d data and 2d projections
    df_data, pr_data, df_dims, pr_dims, train_test, data_mean, data_std, dtu, dtu_flat =\
        data_utils.read_3d_data()
    
    print("\n[+] done reading and normalizing data")
    # Getting the number of training and test subjects
    test_depth, test2d = [], []
    for idx, t in enumerate(train_test):
        if t == 0: # test data
            d3d = df_data[dims[idx]:dims[idx+1]]
            td = d3d[:,:, data_utils.DEPTH_COORD]
            test_depth.append(td[:, dtu])

            t2d = d3d[:,:, data_utils.XY_COORD]
            test2d.append(t2d[:, dtu])
            te_subj += dims[idx+1]-dims[idx]

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
      
        n_joints = len(dtu)

        model = create_model(sess, batch_size)
        print("[+] model loaded")
        
        encoder_inputs, decoder_outputs =\
            model.get_all_batches(test2d, test_depth, training=False)
      
        nbatches = len(encoder_inputs)
        print("[*] there are {0} test batches".format(nbatches))

        # predicted depths
        all_predic = []
        for i in range(nbatches):
            enc_in = encoder_inputs[i]
            dec_out = decoder_outputs[i]
            _, _, predic = model.step(sess, enc_in, dec_out, dp, isTraining=False)
          
            # Un-normalize
            data_mean = np.reshape(data_mean, (data_utils.NUM_JOINTS, 3))
            data_std = np.reshape(data_std, (data_utils.NUM_JOINTS, 3))
          
            data_mean_depth = data_mean[:, data_utils.DEPTH_COORD]

            data_std_depth = data_std[:, data_utils.DEPTH_COORD]

            predic = data_utils.unNormalize_batch(predic, data_mean_depth, data_std_depth, dtu)

            assert predic.shape[0] == FLAGS.batch_size
          
            all_predic.append(predic)

        # Put all the predicted depths together
        predic = np.vstack(all_predic)
        print(predic.shape)
        #print(unNorm_ob_data_3d.shape)
        
        #unNorm_ob_data_3d[:,:,data_utils.Z_COORD] = ob_predic[:,data_utils.DF_NON_COXA_FEMUR]
        
        #np.save("saved_structures/predic.npy", unNorm_ob_data_3d)

def main(_):
    if FLAGS.test:
        test()
    else:
        train()

if __name__ == "__main__":
    tf.compat.v1.app.run()
