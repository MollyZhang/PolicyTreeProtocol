import numpy as np
import random

class ALOHA_Q(object):
    """
    ALOHA-Q player as described in Chu et al.
    """


    def __init__(self,
                 N=64,
                 active=True,
                 do_print=False,
                 name=None,
                 t=0,
                 retry_limit=6,
                 frame = 0,
                 alpha = 0.9,
                 gamma = 0.9):
        self.active = active
        self.do_print = do_print
        self.name = name or hex(random.getrandbits(16))[2:]
        self.N = N
        self.W = 1
        self.Q = [0] * self.N
        self.t = t
        self.retry = 0
        self.retry_limit = retry_limit
        self.frame = 0
        self.scheduled_frame = 0
        self.alpha = alpha
        self.gamma = gamma
        self.slot = np.argmax((np.random.rand(self.N) * 1e-10) + self.Q)

    def get_decision(self):
        self.decision = (self.slot == (self.t % self.N)) and (self.frame == self.scheduled_frame)
        return self.decision and self.active


    def set_active(self, b):
        self.active = b

    def get_estimated_num_players(self):
        return self.N

    def get_depth(self):
        return -np.log2(self.N)

    def learn(self, collision=0, used=0, name=None):
        """collision = a collision occurred on the network;
           used = the network slot was used (by us or others)"""

        if self.do_print:
            print("player name:", self.name, "t:", self.t, "t%N:", self.t%self.N)
            print("before update: -----------------")
            print("Q:", self.Q)
            print("highest slot number:", self.slot)
            print("window size:", self.W)
            print("scheduled frame:", self.scheduled_frame)
            print("frame", self.frame)
            print("decision:", self.decision)

        if self.decision:
            if collision:
                self.W *= 2
                r = -1
                self.update_Q(r)
                self.retry += 1
                if self.retry > self.retry_limit:
                    self.retry = 0
                    self.W = 1
                    self.frame = 0
                    self.scheduled_frame = 0
                else:
                    self.scheduled_frame = np.random.randint(self.W)
            if used:
                r = 1
                self.update_Q(r)
                self.retry = 0
                self.W = 1
                self.frame = 0
                self.scheduled_frame = 0


        if self.do_print:
            print("after update: -----------------")
            print("Q:", self.Q)
            print("window size:", self.W)
            print("scheduled frame:", self.scheduled_frame)
            print("")

    def update_Q(self, r):
        """ as show in paper
            Qt+1 = Qt + alpha * (r - Qt)
        """
        old_Q = self.Q[self.t % self.N]
        self.Q[self.t % self.N] = old_Q + self.alpha * (r - old_Q)

    def get_display_name(self):
        return "ALOHA-Q"

    def tick(self):
        self.t += 1
        self.frame = int(self.t/self.N) % self.W
        if self.t % self.N == 0:
            self.slot = np.argmax((np.random.rand(self.N) * 1e-10) + self.Q)
