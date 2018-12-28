import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import time


def pick_top_n(preds, char_size, top_n=5):
    """Choose top_n most possible charactors in predictions"""
    
    # Set all values other that top_n choices to 0
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    
    # Normalization
    p = p / np.sum(p)
    
    # Randomly choose one charactor
    c = np.random.choice(char_size, 1, p=p)[0]
    return c


class CharRNN():
    def __init__(self, num_classes, batch_size=64, num_chars=50, cell_type='LSTM',
                 rnn_size=128, num_layers=2, learning_rate=0.001,
                 grad_clip=5, train_keep_prob=0.5, sampling=False):
        # if not training
        if sampling:
            batch_size, num_chars = 1, 1
        else:
            batch_size, num_chars = batch_size, num_chars

        tf.reset_default_graph()
        
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_chars = num_chars
        self.cell_type = cell_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        self.pred_keep_prob = 1.

        self.inputs_layer()
        self.rnn_layer()
        self.outputs_layer()
        self.my_loss()
        self.my_optimizer()
        self.saver = tf.train.Saver()

    def inputs_layer(self):
        """build the input layer"""
        shape = (self.batch_size, self.num_chars)
        self.inputs = tf.placeholder(tf.int32, shape=shape, name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=shape, name='targets')

        # add keep_prob
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.rnn_inputs = tf.one_hot(self.inputs, self.num_classes)

    def rnn_layer(self):

        if self.cell_type == 'LSTM':
            cell = tf.contrib.rnn.BasicLSTMCell
        elif self.cell_type == 'GRU':
            cell = tf.contrib.rnn.GRUCell

        cells = [cell(self.rnn_size) for _ in range(self.num_layers)]

        rnn = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.MultiRNNCell(cells), output_keep_prob=self.keep_prob)
        self.initial_state = rnn.zero_state(self.batch_size, dtype=tf.float32)
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(rnn, self.rnn_inputs, initial_state=self.initial_state)

    def outputs_layer(self):
        """build the output layer"""
        # concate the output of rnn_cell，example: [[1,2,3],[4,5,6]] -> [1,2,3,4,5,6]
        seq_output = tf.concat(self.rnn_outputs, axis=1)  # tf.concat(concat_dim, values)
        x = tf.reshape(seq_output, [-1, self.rnn_size])

        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_classes], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))

        self.logits = tf.matmul(x, softmax_w) + softmax_b
        self.prob_pred = tf.nn.softmax(self.logits, name='predictions')

    def my_loss(self):
        """calculat loss according to logits and targets"""

        # One-hot coding
        y_one_hot = tf.one_hot(self.targets, self.num_classes)
        y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())

        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)

    def my_optimizer(self):
        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, variables = zip(*adam.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.optimizer = adam.apply_gradients(zip(gradients, variables))

    def train(self, batches, iters):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            counter = 0
            new_state = sess.run(self.initial_state)

            # Train network
            for x, y in batches:
                counter += 1
                start = time.time()
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run(
                        [self.loss, self.final_state, self.optimizer],
                        feed_dict=feed)
                end = time.time()
                if counter % 200 == 0:
                    print('step: {counter} loss: {batch_loss:.4f}')

                if counter >= iters:
                    break

            self.saver.save(sess, f"checkpoints/i{counter}_l{self.rnn_size}_{self.cell_type}_ckpt")

    def sample(self, checkpoint, n_samples, vocab_size, vocab_to_ind, ind_to_vocab, prime):
        # change text into character list
        samples = [c for c in prime]

        input_chars = [vocab_to_ind[s] for s in samples]
        output_chars = []
        
        with tf.Session() as sess:
            self.saver.restore(sess, checkpoint)
            state = sess.run(self.initial_state)
            
            for input_char in input_chars:
                feed = {self.inputs: [[input_char]],
                        self.keep_prob: self.pred_keep_prob,
                        self.initial_state: state}
                state = sess.run(self.final_state, feed_dict=feed)

            output_char = input_chars[-1]
            for _ in range(n_samples):
                feed = {self.inputs: [[output_char]],
                        self.keep_prob: self.pred_keep_prob,
                        self.initial_state : state}
                preds, state = sess.run([self.prob_pred, self.final_state], feed_dict=feed)

                output_char = pick_top_n(preds, vocab_size, 5)
                output_chars.append(ind_to_vocab[output_char])
                
            return prime + ''.join(output_chars)