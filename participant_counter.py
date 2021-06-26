import collections
import random


class ParticipantCounter(object):

    def __init__(self, l=100):
        self.l = l
        self.queue = collections.deque()

    def hit(self):
        self.queue.appendleft(hex(random.getrandbits(32))[2:])
        self._normalize()

    def set(self, s):
        self.queue.appendleft(s)
        self._normalize()

    def count(self):
        return len(set(self.queue) - {None})

    def estimate(self):
        """Returns an estimate of the number of players."""
        # There's at least one player: the node itself.
        return max(1, self.count())

    def _normalize(self):
        if len(self.queue) > self.l:
            self.queue.pop()

    def spy(self, name):
        return self.queue.count(name)/self.l


class TransmissionCounter(object):

    def __init__(self, l=100):
        self.l = l
        self.queue = collections.deque()

    def transmit(self, x):
        """x=1 for transmit, x=0 otherwise."""
        self.queue.appendleft(x)
        if len(self.queue) > self.l:
            self.queue.pop()

    def set_last(self, x):
        self.queue[-1] = x

    def get_bw(self):
        return sum(self.queue) / self.l
