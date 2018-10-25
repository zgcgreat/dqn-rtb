
import numpy as np
import tensorflow as tf

from agent import agent
from rtb_environment import RTB_environment, get_data, single_camp_test, multi_camp_test

if __name__ == '__main__':
    camp_n = ['1458', '2259', '2261', '2821', '2997', '3358', '3386', '3427', '3476']

    train_file_dict, test_file_dict = get_data(camp_n)

    # AGENT parameters
    epsilon_max = 0.9
    epsilon_min = 0.05
    epsilon_decay_rate = 0.0001
    discount_factor = 0.99
    batch_size = 32
    memory_cap = 100000
    update_frequency = 100
    # random_n = 30000
    budget_scaling = 1 / 32
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
        rtb_environment = RTB_environment(camp_data, episode_length, step_length)
        while rtb_environment.data_count > 0:
            ###THESE INITIALIZATIONS NEED CONSIDERATION. HOW SHOULD LAMBDA AND
            ###BUDGET BE INITIALIZED?
            initial_Lambda = np.random.normal(0.001, 0.00005)
            budget = np.random.normal(budget_scaling * camp_data['budget'] * step_length * \
                                      episode_length / camp_data['imp'], camp_data['budget'] * budget_scaling * 1 / 100)

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

            episode_name = 'Episode {} from camp {}'.format(episode_counter, camp_id)
            print('Episode {} from camp {}::: epsilon = {:.2f}, budget = {:.1f}'
                  .format(episode_counter, camp_id, agent.e_greedy_policy.epsilon, budget))
            rtb_environment.episode_result(episode_name)
            episode_counter += 1

        print('CAMP {} has finished'.format(camp_id))
        episode_counter = 1
        for name, item in camp_data.items():
            del item

    del train_file_dict
    test(agent, test_file_dict, budget_scaling)

    sess.close()