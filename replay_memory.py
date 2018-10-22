import numpy as np

class replay_memory:
    """
    This class will define and construct the replay memory, as well as
    contain function which lets us add to and sample from the replay
    memory.
    """

    def __init__(self, memory_cap, batch_size):
        self.memory_cap = memory_cap
        self.batch_size = batch_size
        self.storage = []

    def store_sample(self, sample):
        """
        This function lets us add samples to our replay memory and checks
        whether the replay memory has reached its cap. Every sample has to be
        a tuple of length 5, including the state, the action, the next state,
        the reward and a boolean variable telling us if we've reached a
        terminal state or not.
        """
        if len(self.storage) == self.memory_cap:
            self.storage.pop(0)
            self.storage.append(sample)
        else:
            self.storage.append(sample)

    def get_sample(self):
        """
        This function retrieves a number of samples from the replay memory
        corresponding to the batch_size. Due to subsequent training, we return
        the retrieved samples as separate vectors, matrices and lists (in the
        case of the boolean variables for terminal states).
        """
        if len(self.storage) <= self.batch_size:
            batch_size = len(self.storage)
        else:
            batch_size = self.batch_size

        A = []
        S = np.zeros([batch_size, len(self.storage[0][1])])
        R = np.zeros(batch_size)
        S_prime = np.zeros([batch_size, len(self.storage[0][3])])
        T = []

        random_points = []
        counter = 0

        while counter < batch_size:
            index = np.random.randint(0, len(self.storage))
            if index not in random_points:
                A.append(self.storage[index][0])
                S[counter, :] = self.storage[index][1]
                R[counter] = self.storage[index][2]
                S_prime[counter, :] = self.storage[index][3]
                T.append(self.storage[index][4])

                random_points.append(index)
                counter += 1
            else:
                continue

        return A, S, R, S_prime, T