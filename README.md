# dqn-rtb
This repository aims to replicate the 2018 paper by Wu et al., 'Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertisement', by creating the agent as described in the paper as well as an environment with which the agent can interact.

## The agent
As described in the paper, the agent will employ a deep Q-network (DQN) with a feed-forward neural network with three hidden layers, all using 100 neurons each. It is not specified in the paper which activation functions are used, but as seem to be standard, this implementation uses ReLU activation functions in the hidden layers and linear activations in the output layer. 

## The environment
The processed iPinYou dataset from Du et al. (2017), see https://github.com/manxing-du/cmdp-rtb, is used to create the RTB environment. There is something like ~16 million bids in the training data. The purpose of the environment is to let the agent participate in the actions and regulate the bid scaling from the state-relevant parameters as they are described by Wu et al. (2018).

## 'MountainCar-v0'
There is an implementation of the DQN-agent code for the mountain car environment in the repository. The purpose is mainly to verify that the program actually works and that it's learning from the environment. 
