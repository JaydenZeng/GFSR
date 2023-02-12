import os
import gpu_utils
import sys
import numpy as np

#gpu_utils.setup_one_gpu()

import tensorflow as tf
from datetime import datetime

from data_reader import DataReader
from data_preprocess import cities

from model import GFSR
from model_utils import count_parameters
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# Parameters
# ==================================================
FLAGS = tf.flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

tf.flags.DEFINE_string("checkpoint_dir", 'checkpoints',
                       """Path to checkpoint folder""")
tf.flags.DEFINE_integer("num_checkpoints", 10,
                        """Number of checkpoints to store (default: 1)""")
tf.flags.DEFINE_integer("num_epochs", 80,
                        """Number of training epochs (default: 10)""")
tf.flags.DEFINE_integer("batch_size", 32,
                        """Batch Size (default: 32)""")
tf.flags.DEFINE_integer("display_step", 20,
                        """Display after number of steps (default: 20)""")

tf.flags.DEFINE_float("learning_rate", 0.001,
                      """Learning rate (default: 0.001)""")
tf.flags.DEFINE_float("max_grad_norm", 5.0,
                      """Maximum value for gradient clipping (default: 5.0)""")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      """Probability of keeping neurons (default: 0.5)""")

tf.flags.DEFINE_integer("hidden_dim", 150,
                        """Hidden dimensions of GRU cell (default: 50)""")
tf.flags.DEFINE_integer("att_dim", 300,
                        """Attention dimensions (default: 100)""")
tf.flags.DEFINE_integer("emb_size", 300,
                        """Word embedding size (default: 200)""")
tf.flags.DEFINE_integer("num_images", 5,
                        """Number of images per review (default: 3)""")
tf.flags.DEFINE_integer("num_classes", 5,
                        """Number of classes of prediction (default: 5)""")

tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        """Allow device soft device placement""")

def evaluate(session, dataset, model, loss, label, prediction, summary_op=None):
  global valid_step

  _y = []
  _pred = []
  for reviews, images, labels in dataset:
    feed_dict = model.get_feed_dict(reviews, images, labels)
    _loss, y_true, y_pred = session.run([loss, label,  prediction], feed_dict=feed_dict)



    for i in range(len(y_true)):
      _y.append(y_true[i])
      _pred.append(y_pred[i])

  f1 = f1_score(_y, _pred, average='macro')
  acc = accuracy_score(_y, _pred)

  return acc, f1





def test(session, data_reader, model, loss, label, prediction):
  cur_acc = []
  cur_f1 = []
  for city in cities:
    acc, f1  = evaluate(session, data_reader.read_test_set(city),
                                   model, loss, label, prediction)
    cur_acc.append(acc)
    cur_f1.append(f1)
  return  cur_acc, cur_f1





def train(session, data_reader, model, train_op, loss, label, prediction):
  for reviews, images, labels in data_reader.read_train_set(batch_size=FLAGS.batch_size):
    step, _, _loss, y_true, y_pred = session.run([model.global_step, train_op, loss, label, prediction],
                                       feed_dict=model.get_feed_dict(reviews, images, labels,
                                                                     FLAGS.dropout_keep_prob))
    _f1 = f1_score(y_true, y_pred, average='macro')
    _acc = accuracy_score(y_true, y_pred) 
    if step %50 == 0:
        print ('current loss: loss={}, acc={:.4f}, f1={:.4f}'.format(_loss, _acc, _f1))


def loss_fn(labels, logits):
  onehot_labels = tf.one_hot(labels, depth=FLAGS.num_classes)
  cross_entropy_loss = tf.losses.softmax_cross_entropy(
    onehot_labels=onehot_labels,
    logits=logits
  )
  l2_loss = tf.contrib.layers.apply_regularization(regularizer = tf.contrib.layers.l2_regularizer(0.0001), weights_list = tf.trainable_variables())
  return cross_entropy_loss


def train_fn(loss, global_step):
  trained_vars = tf.trainable_variables()
  count_parameters(trained_vars)

  # Gradient clipping
  gradients = tf.gradients(loss, trained_vars)
  clipped_grads, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
  train_op = optimizer.apply_gradients(zip(clipped_grads, trained_vars),
                                       name='train_op',
                                       global_step=global_step)
  return train_op



def main(_):
  config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
  with tf.Session(config=config) as sess:
    print('\n{} Model initializing'.format(datetime.now()))

    model = GFSR(FLAGS.hidden_dim, FLAGS.att_dim, FLAGS.emb_size, FLAGS.num_images, FLAGS.num_classes)
    loss = loss_fn(model.labels, model.logits)
    train_op = train_fn(loss, model.global_step)    
    label = model.labels
    prediction = tf.argmax(model.logits, axis=-1)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)
    data_reader = DataReader(num_images=FLAGS.num_images, train_shuffle=True)

    print('\n{} Start training'.format(datetime.now()))

    epoch = 0
    all_acc =[]
    all_f1 = []
    while epoch < FLAGS.num_epochs:
      epoch += 1
      print('\n=> Epoch: {}'.format(epoch))

      train(sess, data_reader, model, train_op, loss, label, prediction)

      acc, f1 = test(sess, data_reader, model, loss, label, prediction)
      print ('---------test---------')
      print (acc)
      print (f1)
      all_acc.append(acc)
      all_f1.append(f1)

    print ('----------max---------')
    print (np.max(all_acc, 0))
    print (np.max(all_f1, 0))




if __name__ == '__main__':
  tf.app.run()
