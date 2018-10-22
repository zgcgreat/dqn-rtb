
import numpy as np
import tensorflow as tf

from e_greedy_policy import e_greedy_policy
from deep_q_network import q_estimator
from replay_memory import replay_memory

class agent:
    """
    This class constructs and defines all the properties of the agent. It has
    to contain the DQN, the e-greedy policy and the replay memory, with the
    ability to train the target DQN using the replay memory. It also lets us
    create and maintain a target network, which we will use to train the
    estimator.
    """

    def __init__(self, epsilon_max, epsilon_min, epsilon_decay_rate,
                 discount_factor, batch_size, memory_cap,
                 state_size, action_size, sess):
        """
        We do not include a state in the initialization of the
        agent since this is exogenous. However, we initialize
        both networks and initialize all of their variables
        through tensorflow.
        """
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.memory_cap = memory_cap
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess

        """We also define some environment-related features that can be useful:"""
        self.reward_episode = 0
        self.reward_list = []

        """Now we define the q-estimator that the agent will use, as well as the
        memory and the e-greedy policy:"""
        self.q_estimator = q_estimator(self.state_size, self.action_size, 'q_estimator')
        self.q_target = q_estimator(self.state_size, self.action_size, 'q_target')
        self.e_greedy_policy = e_greedy_policy(self.epsilon_max, self.epsilon_min,
                                               self.epsilon_decay_rate)
        self.replay_memory = replay_memory(self.memory_cap, self.batch_size)

        self.sess.run(self.q_estimator.var_init)
        self.sess.run(self.q_target.var_init)

    def action(self, state):
        """
        This function uses the e-greedy policy defined in the previous class
        to choose an action.
        """
        return self.e_greedy_policy.action(self.sess, state, self.q_estimator)

    def q_learning(self):
        """
        This function uses the replay memory and the DQN to train the
        DQN. We use the target DQN (i.e. self.q_target) to create an
        action-value estimate for the subsequent states. Then, we update
        the specific action-values using the Bellman equation.
        """
        action_list, state_matrix, reward_vector, \
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
        This function copies the weights from the estimator to the target network,
        i.e. from q_estimator to q_target.
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