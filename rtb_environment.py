
import numpy as np
import os
import pandas as pd
import pickle as pickle


class RTB_environment:
    """
    This class will construct and manage the environments which the
    agent will interact with. The distinction between training and
    testing environment is primarily in the episode length.
    """
    def __init__(self, camp_dict, episode_length, step_length):
        """
        We need to initialize all of the data, which we fetch from the
        campaign-specific dictionary. We also specify the number of possible
        actions, the state, the amount of data which has been trained on,
        and so on.
        :param camp_dict: a dictionary containing data on winning bids, ctr
        estimations, clicks, budget, and so on. We copy the data on bids, ctr
        estimations and clicks; then, we delete the rest of the dictionary.
        :param episode_length: specifies the number of steps in an episode
        :param step_length: specifies the number of auctions per step.
        """
        self.camp_dict = camp_dict
        self.data_count = camp_dict['imp']
        # self.ctr_estimation_data = np.array(camp_dict['data']['pctr'])
        # self.winning_bid_data = np.array(camp_dict['data']['winprice'])
        # self.click_data = list(camp_dict['data']['click'])
        # camp_dict = None

        self.episode_dict = {'auctions':0, 'impressions':0, 'click':0, 'cost':0, 'win-rate':0, 'eCPC':0}

        self.actions = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]

        self.step_length = step_length
        self.episode_length = episode_length

        self.Lambda = 1
        self.time_step = 0
        self.budget = 0
        self.n_regulations = 0
        self.budget_consumption_rate = 0
        self.winning_rate = 0
        self.cost = 0
        self.ctr_value = 0
        self.click = 0
        self.termination = True

        self.state = [self.time_step, self.budget, self.n_regulations,
                      self.budget_consumption_rate, self.cost,
                      self.winning_rate, self.ctr_value]

    def get_camp_data(self):
        """
        This function updates the data variables which are then accessible
        to the step-function. This function also deletes data that has already
        been used and, hence, tries to free up space.
        :return: updated data variables (i.e. bids, ctr estimations and clicks)
        """
        if self.data_count < self.step_length:
            ctr_estimations = np.array(self.camp_dict['data'].iloc[:self.data_count, :]['pctr'])
            winning_bids = np.array(self.camp_dict['data'].iloc[:self.data_count, :]['winprice'])
            clicks = list(self.camp_dict['data'].iloc[:self.data_count, :]['click'])

            self.data_count = 0
            return ctr_estimations, winning_bids, clicks
        else:
            ctr_estimations = np.array(
                self.camp_dict['data'].iloc[self.data_count - self.step_length:self.data_count, :]['pctr'])
            winning_bids = np.array(
                self.camp_dict['data'].iloc[self.data_count - self.step_length:self.data_count, :]['winprice'])
            clicks = list(
                self.camp_dict['data'].iloc[self.data_count - self.step_length:self.data_count, :]['click'])

            # self.ctr_estimation_data = self.ctr_estimation_data[self.step_length:]
            # self.winning_bid_data = self.winning_bid_data[self.step_length:]
            # self.click_data = self.click_data[self.step_length:]

            self.data_count -= self.step_length
            return ctr_estimations, winning_bids, clicks

    def reset(self, budget, initial_Lambda):
        """
        This function is called whenever a new episode is initiated
        and resets the budget, the Lambda, the time-step, the termination
        bool, and so on.
        :param budget: the amount of money the bidding agent can spend during
        the period
        :param initial_Lambda: the initial scaling of ctr-estimations to form bids
        :return: initial state, zero reward and a false termination bool
        """
        self.budget = budget
        self.Lambda = initial_Lambda
        self.n_regulations = self.episode_length

        self.budget_consumption_rate = 0
        self.winning_rate = 0
        self.ctr_value = 0
        self.click = 0

        self.state = [self.time_step, self.budget, self.n_regulations,
                      self.budget_consumption_rate, self.cost,
                      self.winning_rate, self.ctr_value]

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
        This function takes an action from the bidding agent (i.e.
        a change in the ctr-estimation scaling, and uses it to compute
        the agent's bids, s.t. it can compare it to the "winning bids".
        If one of the agent's bids exceed a winning bid, it will subtract
        the cost of the impression from the agent's budget, etc, given that
        the budget is not already depleted.
        :param action_index: an index for the list of allowed actions
        :return: a new state, reward and termination bool (if time_step = 96)
        """
        action = self.actions[action_index]
        self.Lambda = self.Lambda*(1 + action)
        ctr_estimations, winning_bids, clicks = self.get_camp_data()

        bids = ctr_estimations*(1/self.Lambda)
        budget = self.budget
        self.click = 0
        self.cost = 0
        self.ctr_value = 0
        self.winning_rate = 0

        for i in range(min(self.data_count, self.step_length)):
            if bids[i] > winning_bids[i] and budget > bids[i]:
                budget -= winning_bids[i]
                self.click += clicks[i]
                self.ctr_value += ctr_estimations[i]
                self.winning_rate += 1 / min(self.data_count, self.step_length)
            else:
                continue

        self.cost = self.budget - budget
        self.episode_dict['auctions'] += min(self.data_count, self.step_length)
        self.episode_dict['impressions'] += self.winning_rate * self.camp_dict['imp']
        self.episode_dict['click'] += self.click
        self.episode_dict['cost'] += self.cost
        self.episode_dict['win-rate'] += self.winning_rate * min(self.data_count, self.step_length) / self.camp_dict['imp']

        self.budget_consumption_rate = (budget - self.budget) / self.budget
        self.budget = budget
        self.n_regulations -= 1
        self.time_step += 1

        if self.data_count == 0:
            self.termination = True

        self.state = [self.time_step, self.budget, self.n_regulations,
                      self.budget_consumption_rate, self.cost,
                      self.winning_rate, self.ctr_value]

        reward = self.ctr_value + self.winning_rate

        return (self.state, reward, self.termination)

    def episode_result(self, episode_name):
        """
        This function takes some "statistics" from the episode and prints
        the results before it resets them for the next episode.
        :param episode_name: a string referencing the episode and campaign
        :return: prints number of auctions, number of impressions won, number of
        actual clicks, winning rate and effective cost per click.
        """
        if self.episode_dict['click'] == 0:
            self.episode_dict['eCPC'] = 0
        else:
            self.episode_dict['eCPC'] = self.episode_dict['cost'] / self.episode_dict['click']
        print(episode_name + '::: Auctions: {}, Impressions: {:.1f}, Clicks: {}, Cost: {:.1f}, Winning rate: {:.2f}, eCPC: {:.2f}'
              .format(self.episode_dict['auctions'], self.episode_dict['impressions'], self.episode_dict['click'],
                      self.episode_dict['cost'], self.episode_dict['win-rate'], self.episode_dict['eCPC']))
        for i in self.episode_dict:
            self.episode_dict[i] = 0

def single_camp_test(agent, camp_id, test_file_dict, budget):
    """
    This function tests a bidding agent on a number of auctions from
    a single campaign and outputs the results, given a certain scaling of the budget
    to allow for variability in testing.
    :param agent: this is the trained DQN-based bidding agent
    :param test_file_dict: a dictionary containing testing data (bids,
    ctr estimations, clicks), budget, and so on from a single campaign.
    :param budget_scaling: a scaling parameter for the budget
    :return:
    """
    agent.e_greedy_policy.epsilon = 0
    number_of_impressions = test_file_dict['imp']
    #budget = test_file_dict['budget'] * budget_scaling
    test_environment = RTB_environment(test_file_dict, number_of_impressions, step_length=96)
    initial_Lambda = 0.00001

    state, reward, termination = test_environment.reset(budget, initial_Lambda)
    while not termination:
        action = agent.action(state)
        next_state, reward, termination = test_environment.step(action)
        state = next_state
    test_environment.episode_result('CAMP ' + camp_id + ' test')


def get_data(camp_n):
    """
    This function extracts data for certain specified campaigns
    from a folder in the current working directory.
    :param camp_n: a list of campaign names
    :return: two dictionaries, one for training and one for testing,
    with data on budget, bids, number of auctions, etc. The different
    campaigns are stored in the dictionaries with their respective names.
    """
    if type(camp_n) != str:
        return 0
        # train_file_dict = {}
        # test_file_dict = {}
        # data_path = os.path.join(os.getcwd(), 'data\\ipinyou-data')
        #
        # for camp in camp_n:
        #     test_data = pd.read_csv(data_path + '\\' + camp + '\\' + 'test.theta.txt',
        #                              header=None, index_col=False, sep=' ',names=['click', 'winprice', 'pctr'])
        #     train_data = pd.read_csv(data_path + '\\' + camp + '\\' + 'train.theta.txt',
        #                              header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr'])
        #     camp_info = pickle.load(open(data_path + '\\' + camp + '\\' + 'info.txt', "rb"))
        #     test_budget = camp_info['cost_test']
        #     train_budget = camp_info['cost_train']
        #     test_imp = camp_info['imp_test']
        #     train_imp = camp_info['imp_train']
        #
        #     train = {'imp':train_imp, 'budget':train_budget, 'data':train_data}
        #     test = {'imp':test_imp, 'budget':test_budget, 'data':test_data}
        #
        #     train_file_dict[camp] = train
        #     test_file_dict[camp] = test
    else:
        data_path = os.path.join(os.getcwd(), 'data\\ipinyou-data')
        test_data = pd.read_csv(data_path + '\\' + camp_n + '\\' + 'test.theta.txt',
                                header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr'])
        train_data = pd.read_csv(data_path + '\\' + camp_n + '\\' + 'train.theta.txt',
                                 header=None, index_col=False, sep=' ', names=['click', 'winprice', 'pctr'])
        camp_info = pickle.load(open(data_path + '\\' + camp_n + '\\' + 'info.txt', "rb"))
        test_budget = camp_info['cost_test']
        train_budget = camp_info['cost_train']
        test_imp = camp_info['imp_test']
        train_imp = camp_info['imp_train']

        train_file_dict = {'imp': train_imp, 'budget': train_budget, 'data': train_data}
        test_file_dict = {'imp': test_imp, 'budget': test_budget, 'data': test_data}

        return train_file_dict, test_file_dict
