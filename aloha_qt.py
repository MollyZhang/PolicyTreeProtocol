import numpy as np
import random

class ALOHA_QT(object):
    """
    This is the implementation of ALOHA-QT protocol as described in this paper:
    https://escholarship.org/uc/item/1pc8d02b
    """

    def __init__(self, name=None, active=True,
                 t=0,
                 max_period_exponent=8,
                 optimality_window=0.95,
                 initial_noise=0.1,
                 initial_transmit=0.25,
                 inc_success=0.2,
                 inc_collision=0.5,
                 inc_potential_collision=0.5,
                 inc_empty=0.2,
                 relinquish=2e-2,
                 do_print=False):
        self.name = name or hex(random.getrandbits(16))[2:]
        # How much below the optimal
        self.optimality_window = optimality_window
        self.active = active
        self.w = None # So we initialize once we have policies.
        self.initial_transmit = initial_transmit
        self.inc_success = inc_success
        self.inc_collision = inc_collision
        self.inc_potential_collision = inc_potential_collision
        self.inc_empty = inc_empty
        self.do_print = do_print
        self.relinquish = relinquish
        self.max_m = max_period_exponent
        # Time counter.
        self.time = t
        # Creates the policies.
        N, K, W = [], [], []
        for m in range(0, max_period_exponent + 1):
            n = 2 ** m
            for k in range(n):
                w = initial_transmit * ((1. - initial_noise) + initial_noise * np.random.random())
                K.append(k)
                N.append(n)
                W.append(w / (1.2 ** m))
        self.W = np.array(W)
        self.K = np.array(K)
        self.N = np.array(N)
        self.num_policies = len(N)
        # self.active_policies are the policies that would like to transmit.
        self.active_policies = np.zeros_like(self.W)
        # self.selected_policies are the policies that are good enough to transmit.
        self.selected_policies = np.zeros_like(self.W)
        self.decision = False


    def get_decision(self):
        # self.active_policies are the policies that would like to transmit.
        self.active_policies = self.time % self.N == self.K
        # self.selected_policies are the policies that are good enough to transmit.
        self.selected_policies = self.W > self.optimality_window
        self.selected_policies[np.argmax(self.W)] = 1
        # We send if there is at least one selected active policy.
        self.decision = self.active * np.sum(self.active_policies * self.selected_policies) > 0
        return self.decision


    def set_active(self, b):
        self.active = b


    def _get_update_factor(self, sign=1, inc_amount=1.):
        """Gets a new weight vector obtained by applying a multiplicative factor
          to the old one.  If sign=1 the factor is >= 1, due to success;
          if sign=-1 then the factor is <= 1 due to collision."""
        randomness = np.random.random(self.num_policies)
        return np.exp(sign * inc_amount * self.active_policies * randomness)


    def learn(self, collision=0, used=0, name=None):
        """collision = a collision occurred on the network;
           used = the network slot was used (by us or others)"""
        if collision:
            # We took part in a collision.
            new_W = self.W * self._get_update_factor(sign=-1, inc_amount=self.inc_collision)
        elif used:
            # Somebody transmitted successfully.
            if self.decision:
                # We transmitted successfully.
                new_W = self.W * self._get_update_factor(sign=1, inc_amount=self.inc_success)
            else:
                # Somebody else transmitted successfully.
                new_W = self.W * self._get_update_factor(sign=-1, inc_amount=self.inc_potential_collision)
        else:
            # Free.
            new_W = self.W * self._get_update_factor(sign=1, inc_amount=self.inc_empty)
        # If we transmitted, we relinquish the slot with small probability.
        if self.decision and np.random.random() < self.relinquish:
            new_W *= (1 - self.active_policies)
        # Redistributes the loss of w to the w vector, in a noisy way.
        new_W = np.minimum(1., new_W)
        new_W_sum = np.sum(new_W)
        W_decrease = np.sum(self.W) - new_W_sum
        if W_decrease > 0 and new_W_sum < self.initial_transmit * self.num_policies:
            # Redistributes in a noisy way.
            inc = np.random.random(self.num_policies)
            inc /= np.sum(inc)
            new_W += inc * W_decrease
            new_W = np.minimum(1., new_W)
        self.W = new_W


    def get_display_name(self):
        return "ALOHA-QT"


    def tick(self):
        self.time += 1
