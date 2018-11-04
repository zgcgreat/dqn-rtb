
import json
import os

from multiprocessing import Pool
from parameter_test import parameter_camp_test

camp_id = ['1458', '2259', '2997', '2821', '3358', '2261', '3386', '3427', '3476']

"""INPUT: parameter_list_element = [camp_id, epsilon_decay_rate, budget_scaling, 
budget_init_variance, initial_Lambda, step_length, learning_rate]"""

"""RETURNS: dictionary with camp ID, parameter list, epsilon at end of training, total budget for testing, 
the total number of auctions, the results (e.g. impressions, clicks, etc), list with budget spending, list with
lambda values, list with boolean values for unimodal test, list with action values."""

if __name__ == '__main__':
    results = []
    initial_Lambda = 0.0001
    budget_scaling = 1/32
    parameter_list = [[camp_id[0], 0.00025, budget_scaling, 0, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00025, budget_scaling, 2500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00025, budget_scaling, 5000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00025, budget_scaling, 7500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00025, budget_scaling, 10000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00025, budget_scaling, 12500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00025, budget_scaling, 15000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.0001, budget_scaling, 0, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.0001, budget_scaling, 2500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.0001, budget_scaling, 5000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.0001, budget_scaling, 7500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.0001, budget_scaling, 10000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.0001, budget_scaling, 12500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.0001, budget_scaling, 15000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000075, budget_scaling, 0, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000075, budget_scaling, 2500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000075, budget_scaling, 5000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000075, budget_scaling, 7500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000075, budget_scaling, 10000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000075, budget_scaling, 12500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000075, budget_scaling, 15000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00005, budget_scaling, 0, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00005, budget_scaling, 2500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00005, budget_scaling, 5000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00005, budget_scaling, 7500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00005, budget_scaling, 10000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00005, budget_scaling, 12500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00005, budget_scaling, 15000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000025, budget_scaling, 0, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000025, budget_scaling, 2500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000025, budget_scaling, 7500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000025, budget_scaling, 10000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000025, budget_scaling, 12500, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.000025, budget_scaling, 15000, initial_Lambda, 1000, 0.001],
                      [camp_id[0], 0.00025, budget_scaling, 0, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00025, budget_scaling, 2500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00025, budget_scaling, 5000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00025, budget_scaling, 7500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00025, budget_scaling, 10000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00025, budget_scaling, 12500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00025, budget_scaling, 15000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.0001, budget_scaling, 0, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.0001, budget_scaling, 2500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.0001, budget_scaling, 5000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.0001, budget_scaling, 7500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.0001, budget_scaling, 10000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.0001, budget_scaling, 12500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.0001, budget_scaling, 15000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000075, budget_scaling, 0, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000075, budget_scaling, 2500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000075, budget_scaling, 5000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000075, budget_scaling, 7500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000075, budget_scaling, 10000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000075, budget_scaling, 12500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000075, budget_scaling, 15000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00005, budget_scaling, 0, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00005, budget_scaling, 2500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00005, budget_scaling, 5000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00005, budget_scaling, 7500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00005, budget_scaling, 10000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00005, budget_scaling, 12500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00005, budget_scaling, 15000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000025, budget_scaling, 0, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000025, budget_scaling, 2500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000025, budget_scaling, 5000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000025, budget_scaling, 7500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000025, budget_scaling, 10000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000025, budget_scaling, 12500, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.000025, budget_scaling, 15000, initial_Lambda, 500, 0.0001],
                      [camp_id[0], 0.00025, budget_scaling, 0, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00025, budget_scaling, 2500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00025, budget_scaling, 5000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00025, budget_scaling, 7500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00025, budget_scaling, 10000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00025, budget_scaling, 12500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00025, budget_scaling, 15000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.0001, budget_scaling, 0, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.0001, budget_scaling, 2500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.0001, budget_scaling, 5000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.0001, budget_scaling, 7500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.0001, budget_scaling, 10000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.0001, budget_scaling, 12500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.0001, budget_scaling, 15000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000075, budget_scaling, 0, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000075, budget_scaling, 2500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000075, budget_scaling, 5000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000075, budget_scaling, 7500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000075, budget_scaling, 10000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000075, budget_scaling, 12500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000075, budget_scaling, 15000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00005, budget_scaling, 0, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00005, budget_scaling, 2500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00005, budget_scaling, 5000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00005, budget_scaling, 7500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00005, budget_scaling, 10000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00005, budget_scaling, 12500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.00005, budget_scaling, 15000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000025, budget_scaling, 0, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000025, budget_scaling, 2500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000025, budget_scaling, 5000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000025, budget_scaling, 7500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000025, budget_scaling, 10000, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000025, budget_scaling, 12500, initial_Lambda, 250, 0.000075],
                      [camp_id[0], 0.000025, budget_scaling, 15000, initial_Lambda, 250, 0.000075]]
    with Pool(32) as p:
        results = p.map(parameter_camp_test, parameter_list)

    file_path = os.path.join(os.getcwd(), 'results')
    current_experiment = 'TEST_04-11-2018'

    counter = 1
    for i in results:
        file_name = file_path + '/' + current_experiment + '_' + str(counter) + '.json'
        json.dump(i, open(file_name, "w"))
        counter += 1