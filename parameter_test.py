
import numpy as np
import tensorflow as tf

from agent import agent
from rtb_environment import RTB_environment, get_data
from drlb_test import drlb_test
from lin_bid_test import lin_bidding_test
from rand_bid_test import rand_bidding_test

#parameter_list = [camp_id, epsilon_decay_rate, budget_scaling, budget_init_variance, initial_Lambda]

def parameter_camp_test(parameter_list):
    """
    This function should take a camp ID, train an agent for that specific campaign
    and then test the agent for that campaign. We start by defining the hyper-parameters.
    It (currently) takes the whole campaign as an episode.
    """

    epsilon_max = 0.9
    epsilon_min = 0.05
    discount_factor = 1
    batch_size = 32
    memory_cap = 100000
    update_frequency = 100
    episode_length = 96

    camp_id = parameter_list[0]
    budget_scaling = parameter_list[1]
    initial_Lambda = parameter_list[2]
    epsilon_decay_rate = parameter_list[3]
    budget_init_var = parameter_list[4] * budget_scaling
    step_length = parameter_list[5]
    learning_rate = parameter_list[6]
    seed = parameter_list[7]


    action_size = 7
    state_size = 5
    tf.reset_default_graph()
    np.random.seed(seed)
    tf.set_random_seed(seed)
    sess = tf.Session()
    rtb_agent = agent(epsilon_max, epsilon_min, epsilon_decay_rate,
                  discount_factor, batch_size, memory_cap,
                  state_size, action_size, learning_rate, sess)

    camp_n = ['1458', '2259', '2997', '2821', '3358', '2261', '3386', '3427', '3476']
    train_file_dict, test_file_dict = get_data(camp_n)
    test_file_dict = test_file_dict[camp_id]
    total_budget = 0
    total_impressions = 0
    global_step_counter = 0

    for i in camp_n:
        rtb_environment = RTB_environment(train_file_dict[i], episode_length, step_length)
        total_budget += train_file_dict[i]['budget']
        total_impressions += train_file_dict[i]['imp']
        while rtb_environment.data_count > 0:
            episode_size = min(episode_length * step_length, rtb_environment.data_count)
            budget = train_file_dict[i]['budget'] * min(rtb_environment.data_count, episode_size) \
                     / train_file_dict[i]['imp'] * budget_scaling
            budget = np.random.normal(budget, budget_init_var)

            state, reward, termination = rtb_environment.reset(budget, initial_Lambda)
            while not termination:
                action, _, _ = rtb_agent.action(state)
                next_state, reward, termination = rtb_environment.step(action)

                memory_sample = (action, state, reward, next_state, termination)
                rtb_agent.replay_memory.store_sample(memory_sample)
                rtb_agent.q_learning()
                if global_step_counter % update_frequency == 0:
                    rtb_agent.target_network_update()

                rtb_agent.e_greedy_policy.epsilon_update(global_step_counter)
                state = next_state
                global_step_counter += 1

    epsilon = rtb_agent.e_greedy_policy.epsilon
    budget = total_budget / total_impressions * test_file_dict['imp'] * budget_scaling
    imp, click, cost, wr, ecpc, ecpi, camp_info = drlb_test(test_file_dict, budget, initial_Lambda, rtb_agent,
                                                            episode_length, step_length)
    sess.close()
    lin_bid_result = list(lin_bidding_test(train_file_dict[camp_id], test_file_dict, budget, 'historical'))
    rand_bid_result = list(rand_bidding_test(train_file_dict[camp_id], test_file_dict, budget, 'uniform'))


    result_dict = {'camp_id':camp_id, 'parameters': parameter_list[1:], 'epsilon':epsilon, 'total budget':budget,
                   'auctions': test_file_dict['imp'],
                   'camp_result': np.array([imp, click, cost, wr, ecpc, ecpi]).tolist(), 'budget':camp_info[0],
                   'lambda':camp_info[1], 'unimod':camp_info[2], 'action values':camp_info[3],
                   'lin_bid_result':lin_bid_result, 'rand_bid_result':rand_bid_result}
    return result_dict