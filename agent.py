
import numpy as np
import tensorflow as tf

from e_greedy_policy import e_greedy_policy
from deep_q_network import q_estimator
from replay_memory import replay_memory

class agent:
    """
    This class incorporates the NN-based approximator of the Q-function,
    the replay memory and an e-greedy policy to create the bidding agent.
    The class contains the actual Q-learning and a function which copies
    the weights to the target network.
    """
    def __init__(self, epsilon_max, epsilon_min, epsilon_decay_rate,
                 discount_factor, batch_size, memory_cap,
                 state_size, action_size, learning_rate, sess):
        """
        :param epsilon_max: initial epsilon for the exploration-intensive phase
        :param epsilon_min: epsilon value to which the agent converges over time
        :param epsilon_decay_rate: rate at which the epsilon decays exponentially
        :param discount_factor: discount factor for future rewards
        :param batch_size: size of mini-batches used in Q-learning
        :param memory_cap: maximum size of the replay memory
        :param state_size: dimensionality of the state
        :param action_size: number of possible actions
        :param sess: tensorflow session used to initialize, use and train the networks
        """
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory_cap = memory_cap
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.sess = sess

        self.q_estimator = q_estimator(self.state_size, self.action_size, self.learning_rate, 'q_estimator')
        self.q_target = q_estimator(self.state_size, self.action_size, self.learning_rate, 'q_target')
        self.e_greedy_policy = e_greedy_policy(self.epsilon_max, self.epsilon_min, self.epsilon_decay_rate)
        self.replay_memory = replay_memory(self.memory_cap, self.batch_size)

        self.sess.run(self.q_estimator.var_init)
        self.sess.run(self.q_target.var_init)

    def action(self, state):
        """
        :param state: current state in which the agent has to act
        :return: action (index) based on e-greedy policy and estimated future returns
        """
        unimod_test_val = int(self.e_greedy_policy.unimodal_check(self.sess, state, self.q_estimator))
        action_values = self.q_estimator.predict_single(self.sess, state)
        return self.e_greedy_policy.action(self.sess, state, self.q_estimator), unimod_test_val, list(action_values)

    def q_learning(self):
        """
        :return: updated (trained) local network using the target network, discount factor
        and a randomized mini-batch of experiences from the replay memory.
        """
        action_list, state_matrix, reward_vector,\
        next_state_matrix, termination_list = self.replay_memory.get_sample()

        current_q = self.q_estimator.predict_batch(self.sess, state_matrix)
        next_q = self.q_target.predict_batch(self.sess, next_state_matrix)

        for i in range(len(action_list)):
            if termination_list[i] == True:
                current_q[i, action_list[i]] = reward_vector[i]
            else:
                current_q[i, action_list[i]] = reward_vector[i] \
                                               + self.discount_factor * np.amax(next_q[i, :])

        self.q_estimator.train_batch(self.sess, state_matrix, current_q)

    def target_network_update(self, polyak_tau=0.95):
        """
        :param polyak_tau: parameter which regulates the degree to which the
        target network should be updated using local network, e.g.
        polyak_tau = 1 means complete copying of weights from local network
        to target network.
        :return: updated target network with weights from local network
        """
        estimator_params = [t for t in tf.trainable_variables() if \
                            t.name.startswith(self.q_estimator.scope)]
        estimator_params = sorted(estimator_params, key=lambda v: v.name)
        target_params = [t for t in tf.trainable_variables() if \
                         t.name.startswith(self.q_target.scope)]
        target_params = sorted(target_params, key=lambda v: v.name)

        update_ops = []

        for e_v, t_v in zip(estimator_params, target_params):
            op = t_v.assign(polyak_tau * e_v + (1 - polyak_tau) * t_v)
            update_ops.append(op)

        self.sess.run(update_ops)