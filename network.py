import numpy as np
import matplotlib.pyplot as plt

class Network(object):

    def __init__(self, players=[], tdmas=[], l16s=[]):
        self.tdmas = tdmas
        self.set_l16s(l16s)
        self.players = players
        self.history = [] 
        self.reset_counters()

    def __repr__(self):
        s = ''.join(self.history)
        return s

    def set_tdmas(self, tdma_list):
        self.tdmas = tdma_list

    def set_l16s(self, l16_list):
        self.l16s = l16_list # index desinates channel, max three items in list
        assert(len(l16_list)<=3)

    def _tick(self):
        for p in self.players:
            p.tick()
        for t in self.tdmas:
            t.tick()
        for l in self.l16s:
            l.tick()
        # print([t.t for t in self.tdmas])

    def reset_counters(self):
        self.slot_counter = 0
        self.player_counter = np.zeros(len(self.players))
        self.collision_counter = 0
        self.tdma_counter = 0
        self.l16_counter = np.zeros(len(self.l16s))

    def get_tdma_utilization(self):
        return self.tdma_counter / self.slot_counter

    def get_l16_utilization(self):
        return self.l16_counter / self.slot_counter

    def get_player_utilization(self):
        return self.player_counter / self.slot_counter

    def get_player_depths(self):
        return [(p.get_depth() if hasattr(p, 'get_depth') else None) for p in self.players]

    def get_estimated_num_players(self):
        return [(p.get_estimated_num_players() if hasattr(p, 'get_estimated_num_players') else None)
                for p in self.players]

    def get_player_labels(self):
        """Gets the display name for each player."""
        return [p.display_name() for p in self.players]

    def get_collisions(self):
        return self.collision_counter / self.slot_counter

    def plot_w(self):
        fig, axes = plt.subplots()
        for i, p in enumerate(self.players):
            axes.plot(p.w, label="player {}".format(i))
        plt.legend(loc='center left', bbox_to_anchor=(1., 0.5))
        plt.show()

    def round(self):
        """Performs one round of the simulation."""
        self.slot_counter += 1
        # Gets TDMA, L16 and player decisions
        tdmas = np.array([t.transmit() for t in self.tdmas])
        l16s = np.array([l.transmit() for l in self.l16s])
        moves = np.array([p.get_decision() for p in self.players])
        # Computes outcome
        num_tdmas = np.sum(tdmas)
        num_l16s = np.sum(l16s)
        num_players = np.sum(moves)
        total = num_tdmas + num_players + num_l16s
        collision = total > 1
        used = total == 1
        active_name = None
        if used:
            if num_tdmas > 0:
                active_idx = np.argmax(tdmas)
                active_name = self.tdmas[active_idx].name
            elif num_l16s > 0:
                active_idx = np.argmax(l16s)
                active_name = self.l16s[active_idx].name
            else:
                active_idx = np.argmax(moves)
                active_name = self.players[active_idx].name
        # print("T: {} P: {} C: {} U: {}".format(num_tdmas, num_players, collision, used))
        # The players are given feedback.
        for p in self.players:
            p.learn(collision=collision, used=used, name=active_name)
        # We keep statistics.
        if collision:
            self.collision_counter += 1
            self.history.append('C')
        else:
            if num_tdmas > 0:
                self.tdma_counter += 1
                self.history.append('T')
            if num_l16s > 0:
                self.history.append('L')
            self.player_counter += moves
            self.l16_counter += l16s
            if num_players > 0:
                self.history.append(active_name)
            else:
                self.history.append('_')
        self._tick()
