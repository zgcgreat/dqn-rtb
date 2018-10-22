
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import pickle as pickle

from agent import agent

class RTB_train_environment:
    """
    This class should create the training environment. We define the actions,
    states, etc. The class should also track the budget, the budget
    consumption rate, the winning rate, and other state-relevant
    parameters. We also want the class to handle the data.
    """
    def __init__(self, train_camp_dict, episode_length, step_length):
        """
        The initialization should take all of the data from a campaign.
        We use this type of input in order to be able to separate
        the campaigns later so that we can train the agent properly.
        We also input the episode length in order to create similar
        a similar environment for training later on. We include the
        step length to be able to test how the agent behaves with
        different step lengths.
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
        self.winning_rate = 0
        self.ctr_value = 0
        self.click = 0
        self.termination = True

        """We also initialize a dictionary in which we can save the
        performance during an episode:"""
        self.episode_dict = {'auctions':0, 'impressions':0, 'click':0, 'cost':0, 'win-rate':0, 'eCPC':0}

        """We also have to define the space of allowed actions:"""
        self.actions = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]

        """We also have to define the length of a time-step and the length
        of an episode:"""
        self.step_length = step_length
        self.episode_length = episode_length
        """This means that any episode will contain a number of bids corresponding
        to the product of the step length and the episode length. This will be important
        to consider as we will want to partition the campaigns into episodes or
        sub-campaigns during training, meaning that we have to partition the budget and
        split up the impressions."""

        """Finally, we have to initialize the Lambda:"""
        self.Lambda = 1

        """and the state:"""
        self.state = [self.time_step, self.budget, self.n_regulations,
                      self.budget_consumption_rate, self.click,
                      self.winning_rate, self.ctr_value,
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
        self.ctr_value = 0
        self.click = 0

        self.state = [self.time_step, self.budget, self.n_regulations,
                      self.budget_consumption_rate, self.click,
                      self.winning_rate, self.ctr_value,
                      self.Lambda]

        reward = 0
        self.termination = False
        self.time_step = 0

        return (self.state, reward, self.termination)

    # def reward_function(self, ctr, clicks):
    #     """
    #     Should try to create a reward function which weighs ctr and clicks,
    #     i.e. such that both are valued in some way??? Or just weigh with
    #     w = 1 for both, such that a click is like ctr = 1 + ctr_est.
    #     """
    #     return self.winning_value + 1 / self.eCPC
    ###CONSIDER CASE WHERE eCPC = 0 BCZ NO CLICKS WON!


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
        self.click = 0
        self.ctr_value = 0
        self.winning_rate = 0

        for i in range(min(self.imp_index, self.step_length)):
            if bids[i] > winning_bids[i] and budget > bids[i]:
                budget -= winning_bids[i]
                self.click += clicks[i]
                self.ctr_value += ctr_estimations[i]
                self.winning_rate += 1 / self.step_length
            else:
                continue
        del bids, winning_bids, ctr_estimations

        self.episode_dict['auctions'] += min(self.imp_index, self.step_length)
        self.episode_dict['impressions'] += self.winning_rate*self.step_length
        self.episode_dict['click'] += self.click
        self.episode_dict['cost'] += self.budget - budget
        self.episode_dict['win-rate'] += self.winning_rate / self.episode_length

        self.budget_consumption_rate = (budget - self.budget)/self.budget
        self.budget = budget
        self.n_regulations -= 1
        self.time_step += 1

        if self.time_step == self.episode_length:
            self.termination = True

        self.state = [self.time_step, self.budget, self.n_regulations,
                      self.budget_consumption_rate, self.click,
                      self.winning_rate, self.ctr_value,
                      self.Lambda]

        reward = self.ctr_value

        return (self.state, reward, self.termination)

    def episode_result(self, episode_name):
        """
        This function should output the contents of the dictionary
        with the episode performance. It should then reset the
        dictionary for the next episode.
        """
        if self.episode_dict['click'] == 0:
            self.episode_dict['eCPC'] = 0
        else:
            self.episode_dict['eCPC'] = self.episode_dict['cost'] / self.episode_dict['click']
        print(episode_name + '::: Auctions: {}, Impressions: {}, Clicks: {}, Cost: {}, Winning rate: {:.2f}, eCPC: {:.2f}'
              .format(self.episode_dict['auctions'], self.episode_dict['impressions'], self.episode_dict['click'],
                      self.episode_dict['cost'], self.episode_dict['win-rate'], self.episode_dict['eCPC']))
        for i in self.episode_dict:
            self.episode_dict[i] = 0

def test(agent, test_file_dict, budget_scaling):
    """
    This function takes an agent and all of the testing data to create
    an RTB environment with a single episode for every campaign in the
    testing data. It then lets the agent interact with the environments
    and outputs the performance.
    """
    agent.e_greedy_policy.epsilon = 0

    for camp_id, camp_data in test_file_dict.items():
        number_of_impressions = camp_data['imp']
        budget = camp_data['budget']*budget_scaling
        test_environment = RTB_train_environment(camp_data, number_of_impressions, 96)
        initial_Lambda = np.random.normal(0.001, 0.0001)

        state, reward, termination = test_environment.reset(budget, initial_Lambda)
        while not termination:
            action = agent.action(state)
            next_state, reward, termination = test_environment.step(action)
            state = next_state

        test_environment.episode_result(camp_id)
        for name, item in camp_data.items():
            del item


def get_data(camp_n):
    """
    This function fetches all of the data to be used for training and testing.
    It takes as input one or several campaigns and extracts the necessary
    data for both training and testing.
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


camp_n = ['1458', '2259', '2261', '2821']
#, '2997', '3358', '3386', '3427', '3476']

train_file_dict, test_file_dict = get_data(camp_n)

#AGENT parameters
epsilon_max = 0.9
epsilon_min = 0.05
epsilon_decay_rate = 0.0001
discount_factor = 0.99
batch_size = 32
memory_cap = 100000
update_frequency = 100
#random_n = 30000
budget_scaling = 1/32
episode_length = 1000
step_length = 96

action_size = 7
state_size = 8

tf.reset_default_graph()
sess = tf.Session()

agent = agent(epsilon_max, epsilon_min, epsilon_decay_rate,
              discount_factor, batch_size, memory_cap,
              state_size, action_size, sess)

episode_counter = 1
global_step_counter = 0

for camp_id, camp_data in train_file_dict.items():
    rtb_environment = RTB_train_environment(camp_data, episode_length, step_length)
    while rtb_environment.imp_index > 0:

        ###THESE INITIALIZATIONS NEED CONSIDERATION. HOW SHOULD LAMBDA AND
        ###BUDGET BE INITIALIZED?
        initial_Lambda = np.random.normal(0.001, 0.0001)
        budget = np.random.normal(budget_scaling * camp_data['budget'] * step_length * \
                 episode_length / camp_data['imp'], camp_data['budget'] * budget_scaling * 1/100)

        state, reward, termination = rtb_environment.reset(budget, initial_Lambda)
        while not termination:
            action = agent.action(state)
            next_state, reward, termination = rtb_environment.step(action)

            memory_sample = (action, state, reward, next_state, termination)
            agent.replay_memory.store_sample(memory_sample)

            agent.q_learning()
            if global_step_counter % update_frequency == 0:
                agent.target_network_update()

            agent.e_greedy_policy.epsilon_update(global_step_counter)
            state = next_state

            global_step_counter += 1

        episode_name = 'EPISODE {} from CAMP {}'.format(episode_counter, camp_id)
        print('EPISODE {} from CAMP {}::: epsilon = {:.2f}, budget = {:.2f}'
              .format(episode_counter, camp_id, agent.e_greedy_policy.epsilon, budget))
        rtb_environment.episode_result(episode_name)
        episode_counter += 1

    print('CAMP {} has finished'.format(camp_id))
    for name, item in camp_data.items():
        del item


test(agent, test_file_dict, budget_scaling)

sess.close()