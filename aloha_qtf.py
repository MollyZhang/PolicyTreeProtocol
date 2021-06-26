from aloha_qt import ALOHA_QT
from participant_counter import ParticipantCounter
import numpy as np

class QTF(ALOHA_QT):
    """ 
        This is the ALOHA-QTF protocol as described in this paper:
        https://escholarship.org/uc/item/1pc8d02b 
    """
    def __init__(self, name=None, active=True, do_print=False,
                 inc_empty=0.5, relinquish=0.02, mpe=8):
        super().__init__(name=name, do_print=do_print, active=active,
                         relinquish=relinquish, inc_empty=inc_empty,
                         max_period_exponent=mpe)
        self.participants = ParticipantCounter(l=2**self.max_m)
        self.num_players = 1
        self.requested_bandwidth = 1
        self.fair_bandwidth = 1

    def get_estimated_num_players(self):
        return self.num_players

    def _get_bandwidth(self):
        """Gets the total bandwidth used by the policy."""
        bw = 0
        policies = []
        for i, p in enumerate(self.selected_policies):
            if p:
                # We check whether it's a children of a selected policy.
                k, n = self.K[i], self.N[i]
                is_sub = False
                for (other_k, other_n) in policies:
                    if n > other_n and k % other_n == other_k:
                        is_sub = True
                        break
                if not is_sub:
                    # Increments the bw
                    bw += 1./n
                    policies.append((k, n))
        assert 0 < bw <= 1
        return bw


    def learn(self, collision=0, used=0, name=None):
        """collision = a collision occurred on the network;
           used = the network slot was used (by us or others)"""

        if self.do_print:
            print("\nlearning of player", self.name)

        # Gets the estimated number of players, bw and bw_target
        self.num_players = self.participants.estimate() * 1.0
        self.requested_bandwidth = self._get_bandwidth()
        self.fair_bandwidth = 1. / self.num_players

        if collision:
            self.participants.hit()
            f = self._get_update_factor(sign=-1, inc_amount=self.inc_collision)
        elif used:
            # Somebody transmitted successfully.
            self.participants.set(name)
            if self.decision:
                # We transmitted successfully.
                f = self._get_update_factor(sign=1, inc_amount=self.inc_success)
            else:
                # Somebody else transmitted successfully.
                f = self._get_update_factor(sign=-1, inc_amount=self.inc_potential_collision)
        else:
            # Free.
            self.participants.set(None)
            f = self._get_update_factor(sign=1, inc_amount=self.inc_empty)
        new_w = self.W * f
        # relinquish the slot with a small probability.
        if self.decision and np.random.random() < self.relinquish:
            if self.requested_bandwidth > self.fair_bandwidth:
                new_w *= (1 - self.active_policies)

        new_w = np.minimum(1., new_w)
        # Redistributes the loss of w to the w vector, in a noisy way.
        new_w_sum = np.sum(new_w)
        w_decrease = np.sum(self.W) - new_w_sum

        if w_decrease > 0 and new_w_sum < self.initial_transmit * self.num_policies:
            # Redistributes in a noisy way.
            inc = np.random.random(self.num_policies)
            inc = inc / np.sum(inc)
            new_w += inc * w_decrease
            new_w = np.minimum(1., new_w)
        self.W = new_w


    def _get_update_factor(self, sign=1, inc_amount=1.):
        """
        Fair modification to update factor.
        """
        if sign > 0:
            f = 1 - (self.requested_bandwidth / self.fair_bandwidth) ** 2.
        else:
            f = (self.requested_bandwidth / self.fair_bandwidth) ** 0.5
        f = max(0, min(1, f))
        randomness = np.random.random(self.num_policies)
        return np.exp(sign * inc_amount * self.active_policies * randomness * f)
