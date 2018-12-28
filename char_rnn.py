import tensorflow as tf
import numpy as np


class CharRNN():
    def __init__(self, char_to_ind, batch_shape, rnn_size, num_layers,
                 learning_rate, grad_clip, predict=False):
        self.char_to_ind = char_to_ind
        self.num_classes = len(char_to_ind)
        self.num_samples, self.num_chars = batch_shape
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip

        if predict:
            self.num_samples, self.num_chars = (1, 1)
            self.checkpoint = tf.train.latest_checkpoint('checkpoints')

        self.build_network()

    def build_network(self):
        tf.reset_default_graph()
        self.build_inputs_layer()
        self.build_rnn_layer()
        self.build_outputs_layer()
        self.build_loss()
        self.build_optimizer()
        self.saver = tf.train.Saver()

    def build_inputs_layer(self):
        """build the input layer"""
        shape = (self.num_samples, self.num_chars)
        self.inputs = tf.placeholder(tf.int32, shape=shape, name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=shape, name='targets')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.rnn_inputs = tf.one_hot(self.inputs, self.num_classes)

    def build_rnn_layer(self):
        cell = tf.contrib.rnn.BasicLSTMCell
        cells = [cell(self.rnn_size) for _ in range(self.num_layers)]
        rnn = tf.contrib.rnn.MultiRNNCell(cells)
        rnn = tf.contrib.rnn.DropoutWrapper(rnn, output_keep_prob=self.keep_prob) # noqa E501

        self.initial_state = rnn.zero_state(self.num_samples, dtype=tf.float32)
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(
                rnn, self.rnn_inputs, initial_state=self.initial_state)

    def build_outputs_layer(self):
        """build the output layer"""
        # Concatenate the output of rnn_cell，
        # Example: [[1,2,3],[4,5,6]] -> [1,2,3,4,5,6]
        output = tf.concat(self.rnn_outputs, axis=1)
        x = tf.reshape(output, [-1, self.rnn_size])

        with tf.variable_scope('softmax'):
            shape = [self.rnn_size, self.num_classes]
            w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            b = tf.Variable(tf.zeros(self.num_classes))

        self.logits = tf.matmul(x, w) + b
        self.prob_pred = tf.nn.softmax(self.logits, name='predictions')

    def build_loss(self):
        """calculate loss according to logits and targets"""

        # One-hot coding
        y_one_hot = tf.one_hot(self.targets, self.num_classes)
        y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())

        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                          labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)

    def build_optimizer(self):
        adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gradients, variables = zip(*adam.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip)
        self.optimizer = adam.apply_gradients(zip(gradients, variables))

    def train(self, batches, iters, keep_prob=0.5):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            new_state = sess.run(self.initial_state)

            for i in range(iters):
                x, y = next(batches)
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: keep_prob,
                        self.initial_state: new_state}
                loss, new_state, _ = sess.run([self.loss,
                                               self.final_state,
                                               self.optimizer], feed_dict=feed)
                if i % 200 == 0:
                    print(f'step: {i} loss: {loss:.4f}')
                    self.saver.save(sess, f'checkpoints/i{i}_l{self.rnn_size}_ckpt')  # noqa E501

    def sample_top_n(self, preds, top_n=5):
        """Choose top_n most possible characters in predictions"""

        # Set all values other that top_n choices to 0
        p = np.squeeze(preds)
        p[np.argsort(p)[:-top_n]] = 0

        # Normalization
        p = p / np.sum(p)

        # Randomly choose one character
        c = np.random.choice(self.num_classes, 1, p=p)[0]
        return c

    def predict(self, prime, num_char):
        ind_to_char = {v: k for k, v in self.char_to_ind.items()}

        input_chars = [self.char_to_ind[s] for s in list(prime)]
        output_chars = []
        output_char = input_chars[-1]

        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint)
            state = sess.run(self.initial_state)
            # Loop for inputs
            for input_char in input_chars:
                feed = {self.inputs: [[input_char]],
                        self.initial_state: state,
                        self.keep_prob: 1.}
                state = sess.run(self.final_state, feed_dict=feed)

            # Loop for prediction
            for _ in range(num_char):
                feed = {self.inputs: [[output_char]],
                        self.initial_state: state,
                        self.keep_prob: 1.}
                preds, state = sess.run([self.prob_pred, self.final_state],
                                        feed_dict=feed)

                output_char = self.sample_top_n(preds, 5)
                output_chars.append(ind_to_char[output_char])

            return prime + ''.join(output_chars)
