
import numpy as np

class e_greedy_policy:
    """
    This class tracks the decay of the epsilon and uses the current
    epsilon to choose an action.
    """
    def __init__(self, epsilon_max, epsilon_min, epsilon_decay_rate):
        """
        :param epsilon_max: initial epsilon for the exploration-intensive phase
        :param epsilon_min: epsilon value to which the agent converges over time
        :param epsilon_decay_rate: rate at which the epsilon decays exponentially
        """
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay_rate = epsilon_decay_rate
        self.epsilon = self.epsilon_max

    def unimodal_check(self, sess, state, q_estimator):
        """
        This function checks if the array of action-values is unimodal using
        some heuristic tests.
        :param sess: current tf-session used
        :param state: current state in which we want to take an action
        :param q_estimator: network used to estimate the value of certain actions
        :return: boolean variable describing whether the distribution of values
        in the action-value array is unimodal or "abnormal".
        """
        action_values = q_estimator.predict_single(sess, state)
        end = len(action_values)
        i = 1
        if max(action_values) == action_values[0] or max(action_values) == action_values[-1]:
            while i < end and action_values[i-1] > action_values[i]:
                i += 1
            while i < end and action_values[i-1] == action_values[i]:
                i += 1
            while i < end and action_values[i-1] < action_values[i]:
                i += 1
            return i == end
        else:
            while i < end and action_values[i-1] < action_values[i]:
                i += 1
            while i < end and action_values[i-1] == action_values[i]:
                i += 1
            while i < end and action_values[i-1] > action_values[i]:
                i += 1
            return i == end


    def epsilon_update(self, t):
        """
        :param t: current global time-step
        :return: current epsilon
        """
        self.epsilon = max(self.epsilon_max - self.epsilon_decay_rate * t, self.epsilon_min)
        #self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) \
        #               * np.exp(-self.epsilon_decay_rate * t)

    def action(self, sess, state, q_estimator):
        """
        :param sess: current tf-session used
        :param state: current state in which we want to take an action
        :param q_estimator: network used to estimate the value of certain actions
        :return: 0-based index of optimal action or random action
        """
        #if not self.unimodal_check(sess, state, q_estimator) and self.epsilon > 0:
        #    epsilon = max(self.epsilon, 0.5)
        #else:
        #    epsilon = self.epsilon

        if np.random.rand() < self.epsilon:
            return np.random.randint(q_estimator.action_size)
        else:
            action_values = q_estimator.predict_single(sess, state)
            return np.argmax(action_values)