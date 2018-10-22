
import numpy as np

class e_greedy_policy:
    """
    This class tracks the epsilon, contains a function which can carry out
    the policy and choose the actions.
    """

    def __init__(self, epsilon_max, epsilon_min, epsilon_decay_rate):
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = self.epsilon_max
        """We don't include a time step here since we'll be using a global time
        step instead. Also, we don't have to include the action_size since
        this is already a property of the q_estimator which will be used."""

    def epsilon_update(self, t):
        """
        This function calculates the epsilon for a given time step, t.
        """
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) \
                       * np.exp(-self.epsilon_decay_rate * t)

    def action(self, sess, state, q_estimator):
        """
        This function uses the q_estimator and the epsilon to choose an action
        based on the e-greedy policy. The function returns an action index,
        e.g. 0, 1, 2, etc.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(q_estimator.action_size)
        else:
            action_values = q_estimator.predict_single(sess, state)
            return np.argmax(action_values)