# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:38:01 2018

@author: Ostigland
"""

import numpy as np
import tensorflow as tf
import os

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
        
class environment:
    """
    This class should create and describe the environment, e.g. by defining
    the actions, states etc. It should track the budget, the budget 
    consumption rate, the CTR-estimations, winning bids, the KPI-values
    attained in previous time periods etc.
    """
    def __init__(self, ctr_estimations, winning_bids):
        """
        This function should take as inputs the CTR-estimations and the 
        winning bids as vectors. They will make up the core of the 
        environment. 
        """
        self.ctr_estimations = ctr_estimations
        self.winning_bids = winning_bids
        
        """The environment also has to initialize all the state-relevant 
        features:"""
        self.time_step = 0
        self.budget = 0
        self.n_regulations = 0
        self.budget_consumption_rate = 0
        self.cpm = 0
        self.winning_rate = 0
        self.winning_value = 0
        self.termination = True
        
        """We also have to define the space of allowed actions:"""
        self.actions = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
        
        """We also have to define the length of a time-step and the length
        of an episode:"""
        self.step_length = 96
        self.episode_length = 100
        """This means that any episode will contain a number of bids corresponding
        to the product of the step length and the episode length."""
        
        """Finally, we have to initialize the Lambda:"""
        self.Lambda = 0
        
        """and the state:"""
        self.state = [self.time_step, self.budget, self.n_regulations,
                                 self.budget_consumption_rate, self.cpm, 
                                 self.winning_rate, self.winning_value, 
                                 self.Lambda]
        
    def reset(self, budget, initial_Lambda):
        """
        This function should return an initial state, in the same format
        as those in the gym environment, i.e. with some state-relevant
        vector, a numerical reward and a boolean describing whether the
        episode has terminated or not.
        """
        self.budget = budget
        self.Lambda = initial_Lambda
        self.n_regulations = self.episode_length

        self.budget_consumption_rate = 0
        self.cpm = 0
        self.winning_rate = 0
        self.winning_value = 0

        self.state = [self.time_step, self.budget, self.n_regulations,
                                 self.budget_consumption_rate, self.cpm, 
                                 self.winning_rate, self.winning_value, 
                                 self.Lambda]
        
        reward = 0
        """SHOULD WE ALSO CONSIDER COST???"""
        self.termination = False
        self.time_step = 0

        return (self.state, reward, self.termination)        
            
    
    def step(self, action_index):
        """
        This function should describe what happens when a certain action
        takes in a certain state. The function takes as an argument the 
        index of an action, e.g. 0:6 since we have 7 actions.
        """
        action = self.actions[action_index]
        self.Lambda = self.Lambda*(1 + action)
        
        ctr_estimations = self.ctr_estimations[:self.step_length]
        bids = ctr_estimations*(1/self.Lambda)
        winning_bids = self.winning_bids[:self.step_length]
        
        self.winning_rate = 0
        self.winning_value = 0
        self.cpm = 0
        
        new_budget = self.budget
        for i in range(self.step_length):
                if (bids[i] > winning_bids[i]) and (winning_bids[i] < new_budget):
                    new_budget -= winning_bids[i]
                    self.winning_rate += 1/self.step_length
                    self.winning_value += ctr_estimations[i]
                    self.cpm += bids[i]*1000/self.step_length
                    
        self.budget_consumption_rate = (self.budget - new_budget)/self.budget
        self.budget = new_budget            

        self.n_regulations -= 1
        self.time_step += 1
        
        self.ctr_estimations = self.ctr_estimations[self.step_length:]
        self.winning_bids = self.winning_bids[self.step_length:]

        if self.budget < min(winning_bids[self.time_step:]):
            self.termination = True
            #CONSIDER THIS TERMINATION CONDITION. DOES IT AFFECT THE TRAINIG?
            #THAT IS, DOES IT GIVE SKEWED EXPERIENCE???
        
        self.state = [self.time_step, self.budget, self.n_regulations,
                                 self.budget_consumption_rate, self.cpm, 
                                 self.winning_rate, self.winning_value, 
                                 self.Lambda]

        reward = self.winning_value
        return (self.state, reward, self.termination)


###DATA------------------------------------------------------------------------

#os.listdir(...)

camp_n = ['1458', '2259', '2261', '2821', '2997', '3358', '3386', '3427', '3476']
#data_type = ['test.theta', 'train.theta']
training_files = []
#test_files = []

for i in camp_n:
    training_files.append(open(os.path.join\
                            (os.getcwd(), 
                             'iPinYou_data\\train.theta'+'_'+i+'.txt')))
#    test_files.append(open(os.path.join\
#                            (os.getcwd(), 
#                             'iPinYou_data\\test.theta'+'_'+i+'.txt')))
#    info_files.append(open(os.path.join\
#                            (os.getcwd(),
#                             'iPinYou_data\\info'+'_'+i+'.txt')))


data = []

for i in training_files:
    T = i.read().split(' ')
    T.pop(0)
    data.append(T)
    
observation_count = 0
for i in range(len(camp_n)):
    observation_count += len(data[i])/2
    
ctr_estimations = np.zeros(int(observation_count))
winning_bids = np.zeros(int(observation_count))

winning_bids_counter = 0
ctr_estimations_counter = 0
for i in range(len(data)):
    for j in range(len(data[i])):
        if j % 2 == 0:
            winning_bids[winning_bids_counter] = float(data[i][j])
            winning_bids_counter += 1
        else:
            ctr_estimations[ctr_estimations_counter] = float(data[i][j][:-2])
            ctr_estimations_counter += 1



###EXPERIMENTS-----------------------------------------------------------------

#AGENT parameters
epsilon_max = 0.9
epsilon_min = 0.05
epsilon_decay_rate = 0.0001
discount_factor = 0.99
batch_size = 32
memory_cap = 100000
update_frequency = 100
#random_n = 30000
episodes_n = 100000

#ENVIRONMENT parameters
budget = 1000000
initial_Lambda = 0.001

action_size = 7
state_size = 8

tf.reset_default_graph()
sess = tf.Session()


###WE SHOULD CHANGE THIS INITIALIZATION TO ALSO INCLUDE BUDGET INFORMATION,
###AND TEST DATA. WE SHOULD ALSO INCLUDE CAMPAIGN INFO. INPUT
###DICTIONARIES INSTEAD OF LISTS/ARRAYS. THIS WAY, WE CAN MATCH
###EVERY CAMPAIGN WITH ITS BUDGET TO GET PROPER TRAINING
###S.T. RESULTS ARE COMPARABLE TO BENCHMARKS!
rtb_environment = environment(ctr_estimations, winning_bids)

agent = agent(epsilon_max, epsilon_min, epsilon_decay_rate,
              discount_factor, batch_size, memory_cap,
              state_size, action_size, sess)

episode_counter = 1
total_reward = 0
global_step_counter = 0

while episode_counter < episodes_n:
    state = rtb_environment.reset(budget, initial_Lambda)[0]
    termination = False
    episode_reward = 0
    avg_win_rate = 0

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

        global_step_counter += 1
        avg_win_rate += rtb_environment.winning_rate/rtb_environment.episode_length

    total_reward += episode_reward
    ###THIS SHOULD OUTPUT MORE RELEVANT DATA. IT WOULD BE NICE TO KNOW HOW QUICKLY THE BUDGET IS SPENT, E.G:
    ###WE WANT eCPC, CPM, BUDGET LEFT, NUMBER OF AUCTIONS, NUMBER OF IMPRESSIONS WON, ETC.
    print('Episode {} gave reward {} and retained {} in budget out of {} with a win rate of {} and an epsilon {}'
          .format(episode_counter, episode_reward, rtb_environment.budget, budget, avg_win_rate, agent.e_greedy_policy.epsilon))
    episode_counter += 1
    episode_reward = 0


sess.close()