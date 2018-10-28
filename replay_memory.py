import numpy as np

class replay_memory:
    """
    This class will create and manage the replay memory, letting us add
    and extract (batches of) samples.
    """
    def __init__(self, memory_cap, batch_size):
        """
        :param memory_cap: maximum size of the replay memory
        :param batch_size: size of mini-batches sampled from the replay memory
        """
        self.memory_cap = memory_cap
        self.batch_size = batch_size
        self.storage = []
        np.random.seed(1)

    def store_sample(self, sample):
        """
        :param sample: a tuple of state, action, next state, reward and
        termination boolean
        :return: updated replay memory, including the new sample and without
        the "oldest" sample if the memory cap has been reached.
        """
        if len(self.storage) == self.memory_cap:
            self.storage.pop(0)
            self.storage.append(sample)
        else:
            self.storage.append(sample)

    def get_sample(self):
        """
        :return: a mini-batch of samples from the replay memory
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
        np.random.seed(1)
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