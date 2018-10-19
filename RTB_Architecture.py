# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:38:01 2018

@author: Ostigland
"""

import numpy as np
import tensorflow as tf
import os
import pandas as pd
import pickle as pickle

###AGENT-----------------------------------------------------------------------

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

class replay_memory:
    """
    This class will define and construct the replay memory, as well as
    contain function which lets us add to and sample from the replay
    memory.
    """

    def __init__(self, memory_cap, batch_size):
        self.memory_cap = memory_cap
        self.batch_size = batch_size
        self.storage = []

    def store_sample(self, sample):
        """
        This function lets us add samples to our replay memory and checks
        whether the replay memory has reached its cap. Every sample has to be
        a tuple of length 5, including the state, the action, the next state,
        the reward and a boolean variable telling us if we've reached a
        terminal state or not.
        """
        if len(sample) != 5:
            raise Exception('sample has to be a tuple of length 5.')

        if len(self.storage) == self.memory_cap:
            self.storage.pop(0)
            self.storage.append(sample)
        else:
            self.storage.append(sample)

    def get_sample(self):
        """
        This function retrieves a number of samples from the replay memory
        corresponding to the batch_size. Due to subsequent training, we return
        the retrieved samples as separate vectors, matrices and lists (in the
        case of the boolean variables for terminal states).
        """
        if len(self.storage) <= self.batch_size:
            batch_size = len(self.storage)
        else:
            batch_size = self.batch_size

        A = []
        S = np.zeros([batch_size, len(self.storage[0][1])])
        R = np.zeros(batch_size)
        S_prime = np.zeros([batch_size, len(self.storage[0][3])])
        T = []

        random_points = []
        counter = 0

        while counter < batch_size:
            index = np.random.randint(0, len(self.storage))
            if index not in random_points:
                A.append(self.storage[index][0])
                S[counter, :] = self.storage[index][1]
                R[counter] = self.storage[index][2]
                S_prime[counter, :] = self.storage[index][3]
                T.append(self.storage[index][4])

                random_points.append(index)
                counter += 1
            else:
                continue

        return A, S, R, S_prime, T


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

###ENVIRONMENT-----------------------------------------------------------------

class RTB_train_environment:
    """
    This class should create the training environment. We define the actions,
    states, etc. The class should also track the budget, the budget
    consumption rate, the winning rate, and other state-relevant
    parameters. We also want the class to handle the data and the
    campaigns.
    """
    def __init__(self, train_camp_dict):
        """
        The initialization should take all of the bidding data,
        using a dictionary of data files for different campaigns.
        We use this type of input in order to be able to separate
        the campaigns later so that we can train the agent properly.
        We also want to initialize all state-relevant parameters.
        """
        self.train_camp_dict = train_camp_dict
        self.camp_size = self.train_camp_dict['imp']
        self.imp_index = self.camp_size


        """We initialize all state-relevant features:"""
        self.time_step = 0
        self.budget = 0
        self.n_regulations = 0
        self.budget_consumption_rate = 0
        self.cpm = 0
        self.winning_rate = 0
        self.winning_value = 0
        self.imp_won = 0
        self.termination = True

        """We also have to define the space of allowed actions:"""
        self.actions = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]

        """We also have to define the length of a time-step and the length
        of an episode:"""
        self.step_length = 96
        self.episode_length = 1000
        """This means that any episode will contain a number of bids corresponding
        to the product of the step length and the episode length. This will be important
        to consider as we will want to partition the campaigns into episodes or
        sub-campaigns during training, meaning that we have to partition the budget and
        split up the impressions."""

        """Finally, we have to initialize the Lambda:"""
        self.Lambda = 1

        """and the state:"""
        self.state = [self.time_step, self.budget, self.n_regulations,
                      self.budget_consumption_rate, self.cpm,
                      self.winning_rate, self.winning_value,
                      self.Lambda]

    def get_camp_data(self):
        """
        This function should fetch data for training. It should return the
        winnings bids, CTR estimations and clicks.c
        """
        if self.imp_index < self.step_length:
            bidding_data = self.train_camp_dict['data'].iloc[:self.imp_index, :]

            winning_bids = np.array(bidding_data['winprice'])
            ctr_estimations = np.array(bidding_data['pctr'])
            clicks = list(bidding_data['click'])

            self.imp_index = 0

            return winning_bids, ctr_estimations, clicks

        else:
            bidding_data = self.train_camp_dict['data']\
                               .iloc[self.imp_index - self.step_length:self.imp_index, :]

            winning_bids = np.array(bidding_data['winprice'])
            ctr_estimations = np.array(bidding_data['pctr'])
            clicks = list(bidding_data['click'])

            self.imp_index -= self.step_length

            return winning_bids, ctr_estimations, clicks

    def reset(self, budget, initial_Lambda):
        """
        This function should return an initial state from which the agent
        can start interacting with the environment. This means that we need
        an initial Lambda and an initial budget.
        """
        self.budget = budget
        self.Lambda = initial_Lambda
        self.n_regulations = self.episode_length

        self.budget_consumption_rate = 0
        self.winning_rate = 0
        self.winning_value = 0
        self.imp_won = 0

        self.state = [self.time_step, self.budget, self.n_regulations,
                      self.budget_consumption_rate, self.cpm,
                      self.winning_rate, self.winning_value,
                      self.Lambda]

        reward = 0
        self.termination = False
        self.time_step = 0

        return (self.state, reward, self.termination)

    def reward_function(self, ctr, clicks):
        """
        Should try to create a reward function which weighs ctr and clicks,
        i.e. such that both are valued in some way??? Or just weigh with
        w = 1 for both, such that a click is like ctr = 1 + ctr_est.
        """

    def step(self, action_index):
        """
        This function should simulate the agent adjusting the Lambda and
        hence scaling the bids differently. Then, we will plough through
        a number of bids corresponding to self.step_length and return
        the new budget, winning rate, etc.
        """
        action = self.actions[action_index]
        self.Lambda = self.Lambda*(1 + action)

        winning_bids, ctr_estimations, clicks = self.get_camp_data()

        bids = ctr_estimations*(1/self.Lambda)
        budget = self.budget
        self.imp_won = 0
        self.winning_value = 0
        self.winning_rate = 0

        for i in range(min(self.imp_index, self.step_length)):
            if bids[i] > winning_bids[i] and budget > bids[i]:
                budget -= winning_bids[i]
                self.imp_won += clicks[i]
                self.winning_value += ctr_estimations[i]
                self.winning_rate += 1/self.step_length
            else:
                continue

        self.budget_consumption_rate = (budget - self.budget)/self.budget
        self.budget = budget
        self.n_regulations -= 1
        self.time_step += 1

        if self.time_step == self.episode_length:
            self.termination = True

        self.state = [self.time_step, self.budget, self.n_regulations,
                      self.budget_consumption_rate, self.cpm,
                      self.winning_rate, self.winning_value,
                      self.Lambda]

        reward = self.winning_value

        return (self.state, reward, self.termination)


# class RTB_test_environment:
#     """
#     This class should create the testing environment. It
#     should in principal have the same functions as the previous
#     class with the exception that it should also print some
#     benchmarking metrics.
#     """
#     def __init__(self, test_file_dict):

###DATA------------------------------------------------------------------------

#os.listdir(...)

camp_n = ['1458', '2259', '2261', '2821', '2997', '3358']
#, '3386', '3427', '3476']

def get_data(camp_n):
    """
    This function fetches all of the data to be used for training and testing.
    """
    train_file_dict = {}
    test_file_dict = {}
    data_path = os.path.join(os.getcwd(), 'data\\ipinyou-data')

    for camp in camp_n:
        test_data = pd.read_csv(data_path + '\\' + camp + '\\' + 'test.theta.txt',
                                 header=None, index_col=False, sep=' ',names=['click', 'winprice', 'pctr'])
        train_data = pd.read_csv(data_path + '\\' + camp + '\\' + 'train.theta.txt',
                                 header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr'])
        camp_info = pickle.load(open(data_path + '\\' + camp + '\\' + 'info.txt', "rb"))
        test_budget = camp_info['cost_test']
        train_budget = camp_info['cost_train']
        test_imp = camp_info['imp_test']
        train_imp = camp_info['imp_train']

        train = {'imp':train_imp, 'budget':train_budget, 'data':train_data}
        test = {'imp':test_imp, 'budget':test_budget, 'data':test_data}

        train_file_dict[camp] = train
        test_file_dict[camp] = test

    return train_file_dict, test_file_dict

train_file_dict, test_file_dict = get_data(camp_n)

###EXPERIMENTS-----------------------------------------------------------------

#AGENT parameters
epsilon_max = 0.9
epsilon_min = 0.05
epsilon_decay_rate = 0.000025
discount_factor = 0.99
batch_size = 32
memory_cap = 100000
update_frequency = 100
#random_n = 30000

action_size = 7
state_size = 8

tf.reset_default_graph()
sess = tf.Session()

agent = agent(epsilon_max, epsilon_min, epsilon_decay_rate,
              discount_factor, batch_size, memory_cap,
              state_size, action_size, sess)

###TRAINING----------------------------------------------------------------

episode_counter = 1
global_step_counter = 0
episode_reward = 0
total_reward = 0
avg_win_rate = 0
budget_scaling = 1/32
budget_indicator = 0
budget_bool = False

for i in camp_n:
    rtb_environment = RTB_train_environment(train_file_dict[i])
    while rtb_environment.imp_index > 0:
        initial_Lambda = np.random.normal(0.001, 0.0001)
        budget = np.random.normal(budget_scaling*train_file_dict[i]['budget'] * rtb_environment.step_length * \
                 rtb_environment.episode_length / train_file_dict[i]['imp'], 5000)
        state, reward, termination = rtb_environment.reset(budget, initial_Lambda)
        while not termination:
            action = agent.action(state)
            next_state, reward, termination = rtb_environment.step(action)

            episode_reward += reward
            memory_sample = (action, state, reward, next_state, termination)
            agent.replay_memory.store_sample(memory_sample)

            agent.q_learning()
            if global_step_counter % update_frequency == 0:
                agent.target_network_update()

            agent.e_greedy_policy.epsilon_update(global_step_counter)
            state = next_state

            if budget_bool == False:
                if rtb_environment.budget < 10:
                    budget_indicator = rtb_environment.time_step
                    budget_bool = True

            global_step_counter += 1
            avg_win_rate += rtb_environment.winning_rate/rtb_environment.episode_length

        #total_reward += episode_reward
        print('Episode {} gave reward {:.1f} and retained {:.1f} in budget out of {:.2f} with a win rate of {:.2f}'
              ' and budget depletion at {} and an epsilon {:.2f}'
              .format(episode_counter, episode_reward, rtb_environment.budget, budget,
                      avg_win_rate, budget_indicator, agent.e_greedy_policy.epsilon))
        episode_counter += 1
        episode_reward = 0
        avg_win_rate = 0
        budget_bool = False
        budget_indicator = 0

###TEST-----------------------------------------------------------


sess.close()