
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from multiprocessing import Pool

#from agent import agent
from rtb_environment import RTB_environment, get_data
#from drlb_test import drlb_test
from lin_bid_test import lin_bidding_test
from rand_bid_test import rand_bidding_test

def comparison_test(camp_id):
    """
    This function should take a camp ID, train an agent for that specific campaign
    and then test the agent for that campaign. We start by defining the hyper-parameters.
    It (currently) takes the whole campaign as an episode.
    """
    epsilon_max = 0.9
    epsilon_min = 0.05
    epsilon_decay_rate = 0.00005
    discount_factor = 1
    batch_size = 32
    memory_cap = 100000
    update_frequency = 100
    budget_scaling = 1 / 32
    initial_Lambda = 0.0001
    budget_init_var = 5000
    episode_length = 96
    step_length = 500
    learning_rate = 0.0001

    action_size = 7
    state_size = 5
    tf.reset_default_graph()
    np.random.seed(1)
    tf.set_random_seed(1)
    sess = tf.Session()

    #rtb_agent = agent(epsilon_max, epsilon_min, epsilon_decay_rate,
    #              discount_factor, batch_size, memory_cap,
    #              state_size, action_size, learning_rate, sess)
    #rtb_agent.target_network_update()

    camp_n = ['1458', '2259', '2997', '2821', '3358', '2261', '3386', '3427', '3476']
    train_file_dict, test_file_dict = get_data(camp_n)
    test_file_dict = test_file_dict[camp_id]
    total_budget = 0
    total_impressions = 0
    #global_step_counter = 0

    for i in camp_n:
    #     rtb_environment = RTB_environment(train_file_dict[i], episode_length, step_length)
        total_budget += train_file_dict[i]['budget']
        total_impressions += train_file_dict[i]['imp']
    #     while rtb_environment.data_count > 0:
    #         episode_size = min(episode_length * step_length, rtb_environment.data_count)
    #         budget = train_file_dict[i]['budget'] * min(rtb_environment.data_count, episode_size)\
    #                  / train_file_dict[i]['imp'] * budget_scaling
    #         budget = np.random.normal(budget, budget_init_var * budget_scaling)
    #
    #         state, reward, termination = rtb_environment.reset(budget, initial_Lambda)
    #         while not termination:
    #             action, _, _ = rtb_agent.action(state)
    #             next_state, reward, termination = rtb_environment.step(action)
    #
    #             memory_sample = (action, state, reward, next_state, termination)
    #             rtb_agent.replay_memory.store_sample(memory_sample)
    #             rtb_agent.q_learning()
    #             if global_step_counter % update_frequency == 0:
    #                 rtb_agent.target_network_update()
    #
    #             rtb_agent.e_greedy_policy.epsilon_update(global_step_counter)
    #             state = next_state
    #             global_step_counter += 1

    #budget = total_budget / total_impressions * test_file_dict['imp'] * budget_scaling
    budget = total_budget * test_file_dict['imp'] / total_impressions * budget_scaling
    #print(camp_id + ' EPSILON: {} and BUDGET {}'.format(rtb_agent.e_greedy_policy.epsilon, budget))
    #imp, click, cost, wr, ecpc, ecpi, plot_list = drlb_test(test_file_dict, budget, initial_Lambda, rtb_agent, episode_length, step_length)
    #print('DRLB test for camp ' + camp_id + ' ::: Impressions: {}, Clicks: {}, Cost: {}, Win-rate: {:.2f}, eCPC: {:.2f}, '
    #                                        'eCPI: {:.2f}'.format(imp, click, cost, wr, ecpc, ecpi))
    imp, click, cost, wr, ecpc, ecpi = lin_bidding_test(train_file_dict[camp_id], test_file_dict, budget, 'historical')
    print('LIN_BID (historical) test for camp ' + camp_id +
          ' ::: Impressions: {}, Clicks: {}, Cost: {}, Win-rate: {:.2f}, eCPC: {:.2f}, eCPI: {:.2f}'
          .format(imp, click, cost, wr, ecpc, ecpi))
    imp, click, cost, wr, ecpc, ecpi = rand_bidding_test(train_file_dict[camp_id], test_file_dict, budget, 'uniform')
    print('RAND_BID (uniform) test for camp ' + camp_id +
          ' ::: Impressions: {}, Clicks: {}, Cost: {}, Win-rate: {:.2f}, eCPC: {:.2f}, eCPI: {:.2f}'
          .format(imp, click, cost, wr, ecpc, ecpi))
    sess.close()


if __name__ == '__main__':
    camp_n = ['3358', '2259', '2997', '2821', '1458', '2261', '3386', '3427', '3476']
    with Pool(32) as p:
        print(p.map(comparison_test, camp_n))