from sumtree import SumTree
from collections import deque
import random


class ReplayBuffer(object):   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, buffer_size):
        self.tree = SumTree(buffer_size)
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)
        self.num_experiences += 1

    def getBatch(self, batch_size):
        batch = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)