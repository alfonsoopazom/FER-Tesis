#From https://github.com/germain-hug/Deep-RL-Keras/blob/master/utils/memory_buffer.py

import random
import numpy as np

from collections import deque
from ChefsHatGym.Util.SumTree import SumTree

class MemoryBuffer(object):
    """ Memory Buffer Helper class for Experience Replay
    using a double-ended queue or a Sum Tree (for PER)
    """
    def __init__(self, buffer_size, with_per):
        """ Initialization
        """
        if(with_per):
            # Prioritized Experience Replay
            self.alpha = 0.5
            self.epsilon = 0.01
            self.buffer = SumTree(buffer_size)
        else:
            # Standard Buffer
            self.buffer = deque()
        self.count = 0
        self.with_per = with_per
        self.buffer_size = buffer_size

    def memorize(self, state, action, reward, done, new_state, possibleActions,newPossibleActions, error=None):
        """ Save an experience to memory, optionally with its TD-Error"""
        experience = (state, action, reward, done, new_state, possibleActions, newPossibleActions)

        if(self.with_per):
            priority = self.priority(error[0])
            self.buffer.add(priority, experience)
            print(self.buffer)
            self.count += 1
        else:
            # Check if buffer is already full
            if self.count < self.buffer_size:
                self.buffer.append(experience)
                self.count += 1
            else:
                self.buffer.popleft()
                self.buffer.append(experience)

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al.
        """
        return (error + self.epsilon) ** self.alpha

    def size(self):
        """ Current Buffer Occupation
        """
        return self.count

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER) """
        batch = []

        # Sample using prorities
        if(self.with_per):
            T = self.buffer.total() // batch_size
            for i in range(batch_size):
                a, b = T * i, T * (i + 1)
                s = random.uniform(a, b)
                a = self.buffer.get(s)
                size = self.buffer.count()
                idx, error, data = self.buffer.get(s)
                batch.append((*data, idx))

            idx = np.array([i[7] for i in batch])
        # Sample randomly from Buffer
        elif self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)

        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])
        possibleActions = np.array([i[5] for i in batch])
        newPossibleActions = np.array([i[6] for i in batch])

        return s_batch, a_batch, r_batch, d_batch, new_s_batch, possibleActions,newPossibleActions, idx

    def update(self, idx, new_error):
        """ Update priority for idx (PER) """
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        """ Clear buffer / Sum Tree """
        if(self.with_per): self.buffer = SumTree(buffer_size)
        else: self.buffer = deque()
        self.count = 0

class Buffer(object):
    def __init__(self, buffer_size):
        self.alpha = 0.5
        self.epsilon = 0.01
        self.buffer = deque()
        self.count = 0
        self.buffer_size = buffer_size

    def memorize(self, state, action, reward, done, new_state, possibleActions,newPossibleActions, error=None):
        """ Save an experience to memory, optionally with its TD-Error"""
        experience = (state, action, reward, done, new_state, possibleActions, newPossibleActions)

        # Check if buffer is already full
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def priority(self, error):
        """ Compute an experience priority, as per Schaul et al. """
        return (error + self.epsilon) ** self.alpha

    def size(self):
        """ Current Buffer Occupation """
        return self.count

    def update(self, idx, new_error):
        """ Update priority for idx (PER) """
        self.buffer.update(idx, self.priority(new_error))

    def clear(self):
        self.buffer = deque()
        self.count = 0

    def sample_batch(self, batch_size):
        """ Sample a batch, optionally with (PER) """
        batch = []

        # Sample randomly from Buffer
        if self.count < batch_size:
            idx = None
            batch = random.sample(self.buffer, self.count)
            print(batch)
        else:
            idx = None
            batch = random.sample(self.buffer, batch_size)
            print(batch)
        # Return a batch of experience
        s_batch = np.array([i[0] for i in batch])
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        new_s_batch = np.array([i[4] for i in batch])
        possibleActions = np.array([i[5] for i in batch])
        newPossibleActions = np.array([i[6] for i in batch])

        print(s_batch)

        return s_batch, a_batch, r_batch, d_batch, new_s_batch, possibleActions,newPossibleActions, idx