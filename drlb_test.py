
import numpy as np

from rtb_environment import RTB_environment

def drlb_test(test_file_dict, budget, initial_Lambda, agent, episode_length, step_length):
    """
    This function tests a bidding agent on a number of auctions from
    a single campaign and outputs the results.
    :param agent: this is the trained DQN-based bidding agent
    :param test_file_dict: a dictionary containing testing data (bids,
    ctr estimations, clicks), budget, and so on from a single campaign.
    :param budget_scaling: a scaling parameter for the budget
    :return:
    """
    agent.e_greedy_policy.epsilon = 0
    test_environment = RTB_environment(test_file_dict, episode_length, step_length)
    budget_list = []
    Lambda_list = []
    unimod_test_list = []
    action_value_list = []
    episode_budget = 0

    while test_environment.data_count > 0:
        episode_budget = min(episode_length * step_length, test_environment.data_count)\
                         / test_file_dict['imp'] * budget + episode_budget
        state, reward, termination = test_environment.reset(episode_budget, initial_Lambda)
        while not termination:
            action, unimod_test_val, action_value = agent.action(state)
            next_state, reward, termination = test_environment.step(action)
            state = next_state

            budget_list.append(test_environment.budget)
            Lambda_list.append(test_environment.Lambda)
            unimod_test_list.append(unimod_test_val)
            action_value_list.append(action_value)
        episode_budget = test_environment.budget
    impressions, click, cost, win_rate, ecpc, ecpi = test_environment.result()

    return impressions, click, cost, win_rate, ecpc, ecpi, \
           [np.array(budget_list).tolist(), np.array(Lambda_list).tolist(),
            np.array(unimod_test_list).tolist(), np.array(action_value_list).tolist()]