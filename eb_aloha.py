import numpy as np
import random

class EB_ALOHA(object):
    """This class implements 'always-learning' exponential
    backoff ALOHA.  A node learns from collisions and success,
    even when caused by other nodes."""

    def __init__(self,
                 q=0.9,
                 active=True,
                 bias=1.,
                 do_print=False):
        self.active = active
        self.do_print = do_print
        self.delay = 1 # Transmission delay.
        self.name = None or hex(random.getrandbits(16))[2:]
        self.q = q
        self.p = 0.5
        self.bias = bias


    def get_decision(self):
        self.decision = np.random.random() < self.p
        if self.do_print:
            print(self.name, " t:", self.t, "d:", self.delay, "Decision:", self.decision)
        return self.decision and self.active


    def set_active(self, b):
        self.active = b

    def get_estimated_num_players(self):
        return 1. / self.p

    def get_depth(self):
        return -np.log2(self.p)

    def learn(self, collision=0, used=0, name=None):
        """collision = a collision occurred on the network;
           used = the network slot was used (by us or others)"""
        if collision:
            self.p *= (self.q ** self.bias)
        elif used == 0:
            # Good, we transmitted into an empty slot.
            self.p = min(1., self.p / self.q)


    def get_display_name(self):
        return "EB-ALOHA"


    def tick(self):
        pass


