# dqn-rtb
This repository aims to replicate the bidding agent from the 2018 paper by Wu et al., 'Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertisement', by creating the agent as described in the paper as well as an environment inspired by OpenAI's Gym library.

## The agent
The DQN-inspired agent was initially tested in OpenAI's 'MountainCar-v0' environment. In this setting, I employed a simpler architecture using two hidden layers, with 64 and 32 neurons, respectively. This was slow to converge, but it eventually learns how to play the game flawlessly. The code is included in DQN_NN_MountainCar.py. The code for this algorithm was roughly transformed into an attempted replication of the bidding agent constructed by Wu et al. (2018). However, there were some shortcomings in this attempts and the final algorithm incorporates some other elements, all of which are discussed in the thesis.

## Testing
There are three separate testing files for the algorithms, drlb_test.py, lin_bid_test.py and rand_bid_test.py. These were all run in a separate file which employed a multiprocessing library on a 32-core computer, which allowed parallel testing. The files for this are not included in this repository. However, the three test files are sufficient for reproducing the results in the thesis, using seed(1) for both numpy and tensorflow. The parameters are listed in the thesis.
