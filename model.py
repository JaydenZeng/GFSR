import tensorflow as tf
import tensorflow.contrib.rnn as rnn

from data_utils import batch_review_normalize, batch_image_normalize, batch_anps
from layers import bidirectional_rnn, text_attention, visual_aspect_attention
from model_utils import get_shape, load_glove

from data_preprocess import VOCAB_SIZE
from module import ff, multihead_attention, ln, positional_encoding, SigmoidAtt


import numpy as np
import sys
class GFSR:

  def __init__(self, hidden_dim, att_dim, emb_size, num_images, num_classes):
    self.hidden_dim = hidden_dim
    self.att_dim = att_dim
    self.emb_size = emb_size
    self.num_classes = num_classes
    self.num_images = num_images

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

    self.documents = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='reviews')
    self.document_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='review_lengths')
    self.sentence_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='sentence_lengths')

    self.max_num_words = tf.placeholder(dtype=tf.int32, name='max_num_words')
    self.max_num_sents = tf.placeholder(dtype=tf.int32, name='max_num_sents')

    self.images = tf.placeholder(shape=(None, None, 4096), dtype=tf.float32, name='images')
    self.labels = tf.placeholder(shape=(None), dtype=tf.int32, name='labels')
    self.anps = tf.placeholder(shape=(None, None,  None, None), dtype = tf.int32, name='anps')
    self.probs = tf.placeholder(shape=(None, None, None), dtype = tf.float32, name = 'probs')

    with tf.variable_scope('GFSR'):
      self._init_embedding()
      self._init_word_encoder()
      #self._init_sent_encoder()
      self._init_classifier()

  def _init_embedding(self):
    with tf.variable_scope('embedding'):
      self.embedding_matrix = tf.get_variable(
        name='embedding_matrix',
        shape=[VOCAB_SIZE, self.emb_size],
        initializer=tf.constant_initializer(load_glove(VOCAB_SIZE, self.emb_size)),
        dtype=tf.float32
      )
      self.embedded_inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.documents)
      self.embedded_anps = tf.nn.embedding_lookup(self.embedding_matrix, self.anps)
    #  print (np.shape(self.embedded_anps))
    #  sys.exit()

  def _init_word_encoder(self):
    with tf.variable_scope('word') as scope:
      word_rnn_inputs = tf.reshape(
        self.embedded_inputs,
        [-1, self.max_num_words, self.emb_size]
      )
      sentence_lengths = tf.reshape(self.sentence_lengths, [-1])

      # word encoder
      cell_fw = rnn.GRUCell(self.hidden_dim)
      cell_bw = rnn.GRUCell(self.hidden_dim)

      init_state_fw = tf.tile(tf.get_variable('init_state_fw',
                                              shape=[1, self.hidden_dim],
                                              initializer=tf.constant_initializer(1.0)),
                              multiples=[get_shape(word_rnn_inputs)[0], 1])
      init_state_bw = tf.tile(tf.get_variable('init_state_bw',
                                              shape=[1, self.hidden_dim],
                                              initializer=tf.constant_initializer(1.0)),
                              multiples=[get_shape(word_rnn_inputs)[0], 1])

      word_rnn_outputs, _ = bidirectional_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=word_rnn_inputs,
        input_lengths=sentence_lengths,
        initial_state_fw=init_state_fw,
        initial_state_bw=init_state_bw,
        scope=scope
      )
      
      self.word_outputs, self.word_att_weights = text_attention(inputs=word_rnn_outputs,
                                                                att_dim=self.att_dim,
                                                                sequence_lengths=sentence_lengths)
      self.word_outputs = tf.nn.dropout(self.word_outputs, keep_prob=self.dropout_keep_prob)

      sentence_rnn_inputs = tf.reshape(self.word_outputs, [-1, self.max_num_sents, 2 * self.hidden_dim])


      visual_input = tf.layers.dense(self.images, self.att_dim, use_bias=False)
      text_input = tf.layers.dense(sentence_rnn_inputs, self.att_dim, use_bias=False)
      anps_input = tf.layers.dense(self.embedded_anps, self.att_dim, use_bias=False)

      anps_input = tf.reduce_mean(anps_input, 3)
      visual_input = tf.reshape(tf.tile(visual_input, [1, 1, 5]), [tf.shape(visual_input)[0], tf.shape(visual_input)[1], 5, self.att_dim])


      #gated fusion
      norm_visual = tf.nn.l2_normalize(visual_input, -1)
      norm_anps = tf.nn.l2_normalize(anps_input, -1)
      sig_r = tf.nn.sigmoid(tf.layers.dense(tf.concat([norm_visual, norm_anps], -1), self.att_dim, use_bias=False))
      visual_input = tf.multiply(sig_r, norm_visual) + tf.multiply((1-sig_r), norm_anps)


      anp_probs = tf.reshape(tf.tile(self.probs, [1, 1, self.att_dim]), [tf.shape(self.probs)[0], tf.shape(self.probs)[1], 5, self.att_dim])

      visual_input = tf.reduce_mean(tf.multiply(visual_input, anp_probs), 2)

      visual_input = tf.layers.dense(visual_input, self.att_dim, use_bias=False)

      #picture self-attention
      with tf.variable_scope('vv', reuse=tf.AUTO_REUSE):
          enc_vv = multihead_attention(queries=visual_input,
                                       keys=visual_input,
                                       values=visual_input,
                                       num_heads=4,
                                       dropout_rate=0.2,
                                       training=True,
                                       causality=False)
          visual_output = ff(enc_vv, num_units=[4*self.att_dim, self.att_dim])

      #sentence self-attention
      with tf.variable_scope('tt', reuse=tf.AUTO_REUSE):
          enc_tt = multihead_attention(queries=text_input,
                                       keys=text_input,
                                       values=text_input,
                                       num_heads=4,
                                       dropout_rate=0.2,
                                       training=True,
                                       causality=False)
          text_output = ff(enc_tt, num_units=[4*self.att_dim, self.att_dim])

      visual_output =  visual_input
      text_output = text_input

      #visual2text
      with tf.variable_scope('vt', reuse=tf.AUTO_REUSE):
          enc_vt = multihead_attention(queries=visual_output,
                                   keys=text_output,
                                   values=text_output,
                                   num_heads=4,
                                   dropout_rate=0.2,
                                   training = True,
                                   causality=False)
          enc_vt = ff(enc_vt, num_units=[4*self.att_dim, self.att_dim])

      with tf.variable_scope('tv', reuse=tf.AUTO_REUSE):
          enc_tv = multihead_attention(queries=text_output,
                                   keys=visual_output,
                                   values=visual_output,
                                   num_heads=4,
                                   dropout_rate=0.2,
                                   training = True,
                                   causality=False)
          enc_tv = ff(enc_tv, num_units=[4*self.att_dim, self.att_dim])


      #visual2allanps
      all_anps = tf.reshape(anps_input, [tf.shape(anps_input)[0], -1, self.att_dim])

      with tf.variable_scope('va', reuse=tf.AUTO_REUSE):
          enc_va = multihead_attention(queries=visual_output,
                                   keys=all_anps,
                                   values=all_anps,
                                   num_heads=4,
                                   dropout_rate=0.2,
                                   training = True,
                                   causality=False)
          enc_va = ff(enc_va, num_units=[4*self.att_dim, self.att_dim])



      with tf.variable_scope('all_weights', reuse = tf.AUTO_REUSE):
          Wr_wq = tf.get_variable('Wr_wq', [self.att_dim, 1])
          Wm_wq = tf.get_variable('Wm_wq', [self.att_dim, self.att_dim])
          Wu_wq = tf.get_variable('Wu_wq', [self.att_dim, self.att_dim])

          Wr_wa = tf.get_variable('Wr_wa', [self.att_dim, 1])
          Wm_wa = tf.get_variable('Wm_wa', [self.att_dim, self.att_dim])
          Wu_wa = tf.get_variable('Wu_wa', [self.att_dim, self.att_dim])


          Wr_va = tf.get_variable('Wr_va', [self.att_dim, 1])
          Wm_va = tf.get_variable('Wm_va', [self.att_dim, self.att_dim])
          Wu_va = tf.get_variable('Wu_va', [self.att_dim, self.att_dim])


      outputs_vt = SigmoidAtt(enc_vt, Wr_wq, Wm_wq, Wu_wq)
      outputs_tv = SigmoidAtt(enc_tv, Wr_wa, Wm_wa, Wu_wa)
      outputs_va = SigmoidAtt(enc_va, Wr_va, Wm_va, Wu_va)

      self.sentence_outputs = tf.concat([outputs_vt, outputs_tv, outputs_va], -1)
      self.sentence_outputs = tf.nn.dropout(self.sentence_outputs, keep_prob=self.dropout_keep_prob)



  def _init_classifier(self):
    with tf.variable_scope('classifier'):
      self.logits = tf.layers.dense(
        inputs=self.sentence_outputs,
        units=self.num_classes,
        name='logits'
      )

  def get_feed_dict(self, reviews, images, labels, dropout_keep_prob=1.0):
    norm_docs, doc_sizes, sent_sizes, max_num_sents, max_num_words = batch_review_normalize(reviews)
    anps, probs = batch_anps(images, self.num_images)
    fd = {
      self.documents: norm_docs,
      self.document_lengths: doc_sizes,
      self.sentence_lengths: sent_sizes,
      self.max_num_sents: max_num_sents,
      self.max_num_words: max_num_words,
      self.images: batch_image_normalize(images, self.num_images),
      self.labels: labels,
      self.anps: anps,
      self.probs: probs,
      self.dropout_keep_prob: dropout_keep_prob
    }
    return fd
