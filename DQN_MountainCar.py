# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 15:41:50 2018

@author: osksti
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

class q_estimator:
    """
    This class will initialize and build our model, i.e. the DQN. The DQN
    will be a fully-connected feed forward NN (using tensorflow).
    """
    def __init__(self, state_size, action_size, variable_scope):
        self.scope = variable_scope
        self.state_size = state_size
        self.action_size = action_size
        
        #We define the state and action placeholder, i.e. the inputs and targets:
        self.input_pl = tf.placeholder(dtype=np.float32, shape=(None, self.state_size),\
                                       name=self.scope+'input_pl')
        self.target_pl = tf.placeholder(dtype=np.float32, shape=(None, self.action_size),\
                                        name=self.scope+'output_pl')
        
        #We define the network itself:
        self.input_layer = tf.layers.dense(self.input_pl, 32, activation=tf.nn.relu,\
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.random_normal,
                                           name=self.scope+'.input_layer')
        self.hidden_layer = tf.layers.dense(self.input_layer, 32, activation=tf.nn.relu,\
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.random_normal,
                                           name=self.scope+'.hidden_layer')
        self.output_layer = tf.layers.dense(self.hidden_layer, self.action_size,\
                                            activation=None,\
                                           kernel_initializer=tf.initializers.random_normal,
                                           bias_initializer=tf.initializers.random_normal,
                                           name=self.scope+'.output_layer')
        
        #We define the properties of the network:
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
        This function takes a vector of a batch of states and makes predictions
        for all of them.
        """
        return sess.run(self.output_layer, feed_dict={self.input_pl: states})
    
    def train_batch(self, sess, inputs, targets):
        """
        This function takes a batch of examples to train the network. We don't
        return the loss since we can do that using self.loss.
        """
        sess.run(self.optimizer, 
                 feed_dict={self.input_pl: inputs, self.target_pl: targets})
#        print(sess.run(self.loss, 
#                       feed_dict={self.input_pl: inputs, self.target_pl: targets}))
#        if (sess.run(self.loss, 
#                       feed_dict={self.input_pl: inputs, self.target_pl: targets}) > 30):
#            print(inputs)
#            print(targets)

class replay_memory:
    """
    This class will define and construct the replay memory.
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
        if (len(sample) != 5):
            raise Exception('sample has to be a tuple of length 5.')
            
        if (len(self.storage) == self.memory_cap):
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
        if (len(self.storage) <= self.batch_size):
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
        
        while (counter < batch_size):
            index = np.random.randint(0, len(self.storage))
            if (index not in random_points):
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
        #We don't include a time step here since we'll be using a global time
        #step instead. Also, we don't have to include the action_size since
        #this is already a property of the q_estimator which will be used. 
        self.epsilon = self.epsilon_max
        
    def epsilon_update(self, t):
        """
        This function calculates the epsilon for a given time step, t.
        """
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) \
                                            *np.exp(-self.epsilon_decay_rate*t)
    
    
    def action(self, sess, state, q_estimator):
        """
        This function uses the q_estimator and the epsilon to choose an action
        based on the e-greedy policy. The function returns an action index,
        e.g. 0, 1, 2, etc.
        """
        if (np.random.rand() < self.epsilon):
            return np.random.randint(q_estimator.action_size)
        else:
            action_values = q_estimator.predict_single(sess, state)
            return np.argmax(action_values)
        
        
class agent:
    """
    This class constructs and defines all the properties of the agent. It has
    to contain the DQN, the e-greedy policy and the replay memory, with the
    ability to train the target DQN using the replay memory. It should also
    have features allowing it to interact with the environment, storing 
    rewards and so on.
    """
    def __init__(self, epsilon_max, epsilon_min, epsilon_decay_rate, 
                 discount_factor, batch_size, memory_cap,
                 state_size, action_size, sess):
        """
        We do not include a state in the initialization of the 
        agent since this is exogenous. That is, for every episode we'll
        have new states and we'll define a new session each time we run
        the program.
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
        
        #We also define some environment-related features that can be useful:
        self.reward_episode = 0
        self.reward_list = []
        
        #Now we define the q-estimator that the agent will use, as well as the
        #memory and the e-greedy policy:
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
        This function should use the replay memory and the DQN to train the 
        DQN. We use the current DQN (i.e. self.q_estimator) to create an 
        action-value estimate for the subsequent state. Then, we update
        the specific action-values using the Bellman equation.
        """
        action_list, state_matrix, reward_vector,\
        next_state_matrix, termination_list = self.replay_memory.get_sample()
        
        current_q = self.q_estimator.predict_batch(self.sess, state_matrix)
        next_q = self.q_target.predict_batch(self.sess, next_state_matrix)
        
        for i in range(len(action_list)):
            if (termination_list[i] == True):
                current_q[i, action_list[i]] = reward_vector[i]
            else:
                current_q[i, action_list[i]] = reward_vector[i] \
                + self.discount_factor*np.amax(next_q[i, :])
                
        self.q_estimator.train_batch(self.sess, state_matrix, current_q)
        
    def target_network_update(self, polyak_tau=0.95):
        """
        This function copies the weights from the DQN to the target network,
        i.e. from q_estimator to q_target.
        """
        estimator_params = [t for t in tf.trainable_variables() if\
                            t.name.startswith(self.q_estimator.scope)]
        estimator_params = sorted(estimator_params, key=lambda v:v.name)
        target_params = [t for t in tf.trainable_variables() if\
                         t.name.startswith(self.q_target.scope)]
        target_params = sorted(target_params, key=lambda v:v.name)
        
        update_ops = []
        
        for e1_v, e2_v in zip(estimator_params, target_params):
            op = e2_v.assign(polyak_tau*e1_v + (1 - polyak_tau)*e2_v)
            update_ops.append(op)
            
        self.sess.run(update_ops)
    
        
        
###TESTING---------------------------------------------------------------------        

epsilon_max = 1
epsilon_min = 0.01
epsilon_decay_rate = 0.000075
discount_factor = 0.99
batch_size = 32
memory_cap = 20000
update_frequency = 200
episodes_n = 500

tf.reset_default_graph()

env = gym.make('MountainCar-v0')
env.seed(0)

with tf.Session() as sess:
    agent = agent(epsilon_max, epsilon_min, epsilon_decay_rate, 
                 discount_factor, batch_size, memory_cap,
                 env.observation_space.shape[0], env.action_space.n, sess)
    
    
    episode_counter = 1
    global_step_counter = 0
    
    while (episode_counter < episodes_n):
        if (episode_counter % 10 == 0):
            print('Episode {} of {}'.format(episode_counter, episodes_n))

            
        state = env.reset()
        termination = False
        while not termination:
            action = agent.action(state)
            next_state, reward, termination = env.step(action)[:3]
            agent.reward_episode += reward
            
            memory_sample = (action, state, reward, next_state, termination)
            agent.replay_memory.store_sample(memory_sample)
            
            global_step_counter += 1
            agent.q_learning()
            if (global_step_counter % update_frequency == 0):
                agent.target_network_update()
            
            agent.e_greedy_policy.epsilon_update(global_step_counter)
            
        print("Step {},Episode reward: {}, Epsilon: {}"\
              .format(global_step_counter, agent.reward_episode, agent.e_greedy_policy.epsilon))
        
        agent.reward_list.append(agent.reward_episode)
        agent.reward_episode = 0
        episode_counter += 1
        

sess.close()
env.close()

plt.plot(agent.reward_list)plt.plot(agent.reward_list)
