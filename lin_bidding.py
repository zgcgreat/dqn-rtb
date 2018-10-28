
import numpy as np

from rtb_environment import get_data

def lin_bidding_test(camp_id, train_file_dict, test_file_dict, budget_scaling):
    """
    This function takes a specific iPinYou-campaign and evaluates the
    performance of a linear bidding algorithm.
    :param camp_id:
    :return:
    """
    bid_average = np.array(train_file_dict['data']['winprice']).mean()
    ctr_average = np.array(train_file_dict['data']['pctr']).mean()
    test_budget = train_file_dict['budget'] / train_file_dict['imp'] * test_file_dict['imp'] * budget_scaling

    click = list(test_file_dict['data']['click'])
    winning_bids = list(test_file_dict['data']['winprice'])
    ctr_estimations = list(test_file_dict['data']['pctr'])

    impressions = 0
    clicks = 0
    cost = 0
    win_rate = 0
    ecpc = 0

    for i in range(test_file_dict['imp']):
        bid = bid_average * ctr_estimations[i] / ctr_average
        if bid > winning_bids[i] and bid < test_budget - cost:
            impressions += 1
            clicks += click[i]
            cost += winning_bids[i]
            win_rate += 1 / test_file_dict['imp']
        else:
            continue

    if clicks > 0:
        ecpc = cost / clicks

    print('LIN_BID ' + camp_id
        + ' test::: Auctions: {}, Impressions: {:.1f}, Clicks: {}, Cost: {:.1f}, Winning rate: {:.2f}, eCPC: {:.2f}'
        .format(test_file_dict['imp'], impressions, clicks, cost, win_rate, ecpc))