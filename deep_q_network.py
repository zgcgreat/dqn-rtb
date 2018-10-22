
import numpy as np
import tensorflow as tf

class q_estimator:
    """
    This class will initialize and build our model, i.e. the DQN. The DQN
    will be a feed-forward neural network with three hidden layers each
    consisting of 100 neurons. This is the architecture as described
    in the paper.
    """

    def __init__(self, state_size, action_size, variable_scope):
        self.scope = variable_scope
        self.state_size = state_size
        self.action_size = action_size

        """"We define the state and action placeholder, i.e. the inputs and targets:"""
        self.input_pl = tf.placeholder(dtype=np.float32, shape=(None, self.state_size),
                                       name=self.scope + 'input_pl')
        self.target_pl = tf.placeholder(dtype=np.float32, shape=(None, self.action_size),
                                        name=self.scope + 'output_pl')

        """We define the architecture of the network:"""
        self.first_hidden_layer = tf.layers.dense(self.input_pl, 100, activation=tf.nn.relu,
                                                  kernel_initializer=tf.initializers.random_normal,
                                                  bias_initializer=tf.initializers.random_normal,
                                                  name=self.scope + '.first_hidden_layer')
        self.second_hidden_layer = tf.layers.dense(self.first_hidden_layer, 100, activation=tf.nn.relu,
                                                   kernel_initializer=tf.initializers.random_normal,
                                                   bias_initializer=tf.initializers.random_normal,
                                                   name=self.scope + '.second_hidden_layer')
        self.third_hidden_layer = tf.layers.dense(self.second_hidden_layer, 100, activation=tf.nn.relu,
                                                  kernel_initializer=tf.initializers.random_normal,
                                                  bias_initializer=tf.initializers.random_normal,
                                                  name=self.scope + '.third_hidden_layer')
        self.output_layer = tf.layers.dense(self.third_hidden_layer, self.action_size,
                                            activation=None, kernel_initializer=tf.initializers.random_normal,
                                            bias_initializer=tf.initializers.random_normal,
                                            name=self.scope + '.output_layer')

        """We define the properties of the network:"""
        self.loss = tf.losses.mean_squared_error(self.target_pl, self.output_layer)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.var_init = tf.global_variables_initializer()

    def predict_single(self, sess, state):
        """
        This function takes a single state and makes a prediction for it.
        """
        return sess.run(self.output_layer,
                        feed_dict={self.input_pl: np.expand_dims(state, axis=0)})

    def predict_batch(self, sess, states):
        """
        This function takes a batch of states and makes predictions
        for all of them.
        """
        return sess.run(self.output_layer, feed_dict={self.input_pl: states})

    def train_batch(self, sess, inputs, targets):
        """
        This function takes a batch of examples to train the network.
        """
        sess.run(self.optimizer,
                 feed_dict={self.input_pl: inputs, self.target_pl: targets})