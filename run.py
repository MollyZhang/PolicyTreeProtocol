import math

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np

matplotlib.rcParams['figure.figsize'] = (5.0, 2.5)
matplotlib.rcParams['errorbar.capsize'] = 5
params = {'legend.fontsize': 'large',
          'axes.labelsize': 'large',
          'axes.titlesize':'large',
          'xtick.labelsize':'large',
          'ytick.labelsize':'large'}
matplotlib.rcParams.update(params)
plt.rc('font', family='serif')
plt.rc('font', serif='Times')
# plt.style.use('seaborn-whitegrid')
plt.rc('text', usetex=True)

class Run(object):

    def __init__(self, net, frame=100):
        """frame is the length of a frame."""
        self.net = net
        self.tdma_utilization = []
        self.l16_utilization = []
        self.player_utilization = []
        self.collisions = []
        self.empty_incentives = []
        self.kind_incentives = []
        self.player_names = []
        self.depths = []
        self.estimated_n = []
        self.frame = frame
        self.actives = []
        self.stats_prepared = False

    def run_frame(self):
        for j in range(self.frame):
            self.net.round()
        self.tdma_utilization.append(self.net.get_tdma_utilization())
        self.l16_utilization.append(self.net.get_l16_utilization())
        self.player_utilization.append(self.net.get_player_utilization())
        self.actives.append(np.array([p.active for p in self.net.players]))
        self.collisions.append(self.net.get_collisions())
        if len(self.net.players) > 0 and hasattr(self.net.players[0], 'kind_incentive'):
            self.empty_incentives.append(self.net.players[0].empty_incentive)
            self.kind_incentives.append(self.net.players[0].kind_incentive)
        self.depths.append(self.net.get_player_depths())
        self.estimated_n.append(self.net.get_estimated_num_players())
        self.net.reset_counters()

    def plot_net(self):
        self.net.plot_w()


    def prepare_stats(self, stat_len=10, bottom_player_fraction=0.1,
                      plot_fairness=True):
        """name is used to save the images. stat_len indicates how many blocks there are
        in a statistical block.  bottom_fraction is the fraction of players at the bottom
        for which we compute the fair share."""
        # For players, we have a list of arrays, one for each time.
        # We need to produce one line per user.
        self.player_utilization = np.vstack(self.player_utilization)
        self.l16_utilization = np.vstack(self.l16_utilization)
        self.actives = np.vstack(self.actives)
        self.total_utilization = np.sum(self.player_utilization, axis=1)
        self.total_utilization += self.tdma_utilization
        self.total_utilization += np.sum(self.l16_utilization, axis=1)
        collisions = np.array(self.collisions)
        self.empty = 1. - self.total_utilization - collisions
        self.num_times, self.num_players = self.player_utilization.shape
        _, self.num_l16s = self.l16_utilization.shape
        # Computes reduced arrays for stats.
        self.num_stat_times = self.num_times // stat_len
        stat_players = np.zeros((self.num_stat_times, self.num_players))
        stat_actives = np.ones((self.num_stat_times, self.num_players))
        for i in range(self.num_stat_times):
            for j in range(self.num_players):
                stat_players[i, j] = np.average(self.player_utilization[i * stat_len : (i + 1) * stat_len, j])
                stat_actives[i, j] = np.all(self.actives[i * stat_len : (i + 1) * stat_len, j])

        if plot_fairness:
            # Compute bottom 10% vs. fair share and Jain's index.
            self.bottom_fair_ratio = []
            self.mid_fair_ratio = []
            self.jain = []
            for i in range(self.num_stat_times):
                utils = np.array([stat_players[i, j] for j in range(self.num_players) if stat_actives[i, j]])
                utils *= self.frame * stat_len
                tot_util = np.sum(utils)
                num_active_players = len(utils)
                jain_i = tot_util ** 2 / (num_active_players * np.sum(utils * utils))
                self.jain.append(jain_i)
                num_bottom_players = math.ceil(num_active_players * bottom_player_fraction)
                num_mid_players = math.ceil(num_active_players / 2)
                utils.sort()
                if num_bottom_players == 0:
                    self.bottom_fair_ratio.append(None)
                else:
                    bottom_util = np.sum(utils[0:num_bottom_players])
                    fair_util = tot_util * num_bottom_players / num_active_players
                    self.bottom_fair_ratio.append(bottom_util / fair_util)
                if num_mid_players == 0:
                    self.mid_fair_ratio.append(None)
                else:
                    mid_util = np.sum(utils[0:num_mid_players])
                    fair_util = tot_util * num_mid_players / num_active_players
                    self.mid_fair_ratio.append(mid_util / fair_util)
        self.stats_prepared = True


    def plot_stats(self, name=None, caption_players=True, plot_players=True,
                   allactive=True, stat_len=10, bottom_player_fraction=0.1,
                   bw_height=2.5, plot_fairness=True,
                   expand_fairness=False, plot_num_estimate=True):
        """name is used to save the images. stat_len indicates how many blocks there are
        in a statistical block.  bottom_fraction is the fraction of players at the bottom
        for which we compute the fair share."""
        if not self.stats_prepared:
            self.prepare_stats(stat_len=stat_len,
                               bottom_player_fraction=bottom_player_fraction,
                               plot_fairness=plot_fairness)
        # For players, we have a list of arrays, one for each time.
        # We need to produce one line per user.
        matplotlib.rcParams['figure.figsize'] = (5.0, bw_height)
        # Plots utilization.
        fig, ax = plt.subplots()
        ax.plot(self.total_utilization, label='Success', color='black', ls='-')
        ax.plot(self.collisions, label='Collision', color='red', ls=':')
        ax.plot(self.empty, label='Empty', color='darkgreen', ls='--')
        if len(self.net.tdmas) > 0:
            ax.plot(self.tdma_utilization, label='TDMA', color='blue')
        if plot_players:
            colors = iter(cm.summer(np.linspace(0., 0.3, self.num_players)))
            for i in range(self.num_players):
                c = next(colors)
                if caption_players:
                    ax.plot(self.player_utilization[:, i], label='{} {}'.format(
                            self.net.players[i].get_display_name(), i+1), color=c)
                else:
                    ax.plot(self.player_utilization[:, i], color=c)
        if len(self.net.l16s) > 0:
            colors = iter(cm.winter(np.linspace(0., 0.5, self.num_l16s)))
            for i in range(self.num_l16s):
                c = next(colors)
                ax.plot(self.l16_utilization[:, i], label='{} channel {}'.format(
                        self.net.l16s[i].get_display_name(), i), color=c)
        # plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
        ax.grid()
        plt.legend()
        plt.xlabel("Time blocks (1 time block = %d time slots)" % self.frame)
        plt.ylabel("Network utilization")
        plt.ylim((-0.05, 1.05))
        plt.xlim(-1, len(self.total_utilization) + 1)
        if name is not None:
            plt.savefig(name + "_bw.pdf", bbox_inches='tight')
        plt.show()

        # Plots all fairness.
        if plot_fairness:
            fix, ax = plt.subplots()
            ax.plot(self.jain, color='black', label='Jain', ls='-')
            bfr = np.array(self.bottom_fair_ratio)
            ax.plot(bfr, color='green', label='$F_{10\%}$', ls='--')
            plt.ylim(-0.1, 1.1)
            plt.xlim(-1/stat_len, (len(self.total_utilization) + 1) / stat_len)
            ax.grid()
            plt.legend()
            plt.ylabel("Fairness")
            plt.xlabel("Time blocks (1 time block = %d time slots)" % (self.frame * stat_len))
            if name is not None:
                plt.savefig(name + '_fairness.pdf', bbox_inches='tight')
            plt.show()

        if expand_fairness:
        # Plots Jain fairness.
            fig, ax = plt.subplots()
            ax.plot(self.jain, color='black', ls='-')
            plt.ylim(-0.1, 1.1)
            ax.grid()
            plt.title("Jain's fairness index")
            plt.xlabel("Time blocks (1 time block = %d time slots)" % (self.frame * stat_len))
            if name is not None:
                plt.savefig(name + '_jain.pdf', bbox_inches='tight')
            plt.show()

            # Plots bottom 10% vs. fair share.
            fig, ax = plt.subplots()
            ax.plot(bfr, color='black', ls='-')
            ax.grid()
            plt.ylim(-0.1, 1.1)
            # plt.legend()
            plt.title("Fairness wrt bottom "
                      + str(int(bottom_player_fraction * 100))
                     + "% of players")
            plt.xlabel("Time blocks (1 time block = %d time slots)" % (self.frame * stat_len))
            if name is not None:
                plt.savefig(name + '_ratios.pdf', bbox_inches='tight')
            plt.show()

        # Plots number of active nodes.
        if not allactive:
            matplotlib.rcParams['figure.figsize'] = (5.0, 1.2)
            fig, ax = plt.subplots()
            ax.plot(np.sum(self.actives, axis=1), color='black')
            ax.grid()
            ymax = np.max(np.sum(self.actives, axis=1))
            plt.ylim(-1, ymax * 1.2)
            plt.xlim(-1, len(self.total_utilization) + 1)
            plt.ylabel("Active nodes")
            plt.xlabel("Time blocks (1 time block = %d time slots)" % self.frame)
            if name is not None:
                plt.savefig(name + "_numnodes.pdf", bbox_inches='tight')
            plt.show()

        # Plots estimated number of active nodes.
        if plot_num_estimate and self.estimated_n[0][0] is not None:
            matplotlib.rcParams['figure.figsize'] = (5.0, 2)
            fig, ax = plt.subplots()
            average_estimate = [np.mean(i) for i in self.estimated_n]
            ax.plot(average_estimate, color='black')
            ax.grid()
            ymax = np.max(average_estimate)
            plt.ylim(-1, ymax * 1.2)
            plt.xlim(-1, len(self.total_utilization) + 1)
            plt.ylabel("Estimated number of nodes")
            plt.xlabel("Time blocks (1 time block = %d time slots)" % self.frame)
            #if name is not None:
            #    plt.savefig(name + "_numnodes.pdf", bbox_inches='tight')
            plt.show()
