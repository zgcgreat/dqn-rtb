
import numpy as np
import tensorflow as tf

class q_estimator:
    """
    This clas creates and manages the networks which will be incorporated
    into the agent. The structure of the network(s) follow Wu et al. (2018),
    with three hidden layers with 100 neurons each.
    """
    def __init__(self, state_size, action_size, learning_rate, variable_scope):
        """
        :param state_size: the dimensionality of the state, which determines
        the size of the input
        :param action_size: the number of possible actions, which determines
        the size of the output
        :param variable_scope: categorizes the names of the tf-variables for
        the local network and the target network.
        """
        self.scope = variable_scope
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        self.input_pl = tf.placeholder(dtype=np.float32, shape=(None, self.state_size),
                                       name=self.scope + 'input_pl')
        self.target_pl = tf.placeholder(dtype=np.float32, shape=(None, self.action_size),
                                        name=self.scope + 'output_pl')

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
                                            activation=tf.nn.relu, kernel_initializer=tf.initializers.random_normal,
                                            bias_initializer=tf.initializers.random_normal,
                                            name=self.scope + '.output_layer')

        self.loss = tf.losses.mean_squared_error(self.target_pl, self.output_layer)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.95).minimize(self.loss)
        self.var_init = tf.global_variables_initializer()

    def predict_single(self, sess, state):
        """
        :param sess: current tf-session used
        :param state: current state for which we want to estimate the value
        of taking certain actions
        :return: estimated value of taking certain actions
        """
        return sess.run(self.output_layer,
                        feed_dict={self.input_pl: np.expand_dims(state, axis=0)})[0]

    def predict_batch(self, sess, states):
        """
        :param sess: current tf-session used
        :param states: batch of states for which we want to estimate values of
        taking certain actions
        :return: estimated values of taking certain actions in a single tensor
        """
        return sess.run(self.output_layer, feed_dict={self.input_pl: states})

    def train_batch(self, sess, inputs, targets):
        """
        :param sess: current tf-session used
        :param inputs: batch of inputs, i.e. states, for which we want to train our
        network
        :param targets: target values with which we want to train our network,
        i.e. estimated returns from taking certain actions
        :return: updated (trained) network
        """
        sess.run(self.optimizer,
                 feed_dict={self.input_pl: inputs, self.target_pl: targets})