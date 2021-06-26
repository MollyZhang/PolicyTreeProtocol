import numpy as np
import json
from network import Network
from run import Run


class SimpleRun(object):
    pass

def run_to_dict(run):
    return dict(
        utilization = list(run.total_utilization),
        jain = list(run.jain),
        bfr = list(run.bottom_fair_ratio),
        empty = list(run.empty),
        collisions = list(run.collisions),
        num_active = [int(i) for i in run.actives.sum(axis=1)],
    )
    
def dict_to_run(d):
    r = SimpleRun()
    r.total_utilization = np.array(d['utilization'])
    r.jain = np.array(d['jain'])
    r.bottom_fair_ratio = np.array(d['bfr'])
    try: 
        r.empty = np.array(d["empty"])
        r.collisions = np.array(d["collisions"])
    except:
        pass
    try:
        r.num_active = np.array(d['num_active'])
    except:
        pass
    return r
    
def save_runs(runs, fn):
    runs_o = [run_to_dict(r) for r in runs]
    with open(fn, 'w') as f:
        json.dump(runs_o, f)
        
def read_runs(fn):
    with open(fn, 'r') as f:
        runs_o = json.load(f)
        return [dict_to_run(d) for d in runs_o]


def run_n(player_class, num_players=10, 
           do_print=False, seed=0, delayAck=True, slot_per_frame=100, 
           num_frame=100, detect_energy=True, stat_len=10, plot=True):
    np.random.seed(seed)
    if delayAck:
        net = Network(players=[player_class(name=str(i)) for i in range(num_players)], 
                       do_print=do_print, detect_energy=detect_energy)
    else:
        net = Network(players=[player_class(name=str(i)) for i in range(num_players)])
    r = Run(net, frame=slot_per_frame)
    for i in range(num_frame):
        r.run_frame()
    
    if plot:
        r.plot_stats(caption_players=False, plot_players=True, 
             allactive=False, stat_len=stat_len, bw_height=2)
    return net


def reverse_ramp(player_class, do_print=False, seed=0, delayAck=True, 
         slot_per_frame=100, num_frame=100, detect_energy=True,
         stat_len=10, plot=False, **kwargs):
    """
    node number: 50, 10, 40
    """

    np.random.seed(seed)
    if delayAck:
        net = Network(players=[player_class(**kwargs) for i in range(50)], 
                       do_print=do_print, detect_energy=detect_energy)
    else:
        net = Network([player_class(**kwargs) for _ in range(50)])
    r = Run(net, frame=slot_per_frame)
    for pl_idx in range(50):
        net.players[pl_idx].set_active(True)
    for i in range(50):
        r.run_frame()
    for i in range(40):
        net.players[i].set_active(False)
        r.run_frame()
    for i in range(100):
        r.run_frame()
    for i in range(30):
        net.players[i].set_active(True)
        r.run_frame()
    for i in range(100):
        r.run_frame()
    r.prepare_stats()
    if plot:
        r.plot_stats(caption_players=False, plot_players=True, 
                     allactive=False, stat_len=stat_len, bw_height=2)
    return r


def ramp(player_class, do_print=False, seed=0, delayAck=True, 
         slot_per_frame=100, num_frame=100, detect_energy=True,
         stat_len=10, plot=False, **kwargs):
    """
    node number: 10, 50, 30
    """

    np.random.seed(seed)
    if delayAck:
        net = Network(players=[player_class(**kwargs) for i in range(50)], 
                       do_print=do_print, detect_energy=detect_energy)
    else:
        net = Network([player_class(**kwargs) for _ in range(50)])
    r = Run(net, frame=slot_per_frame)
    for pl_idx in range(10, 50):
        net.players[pl_idx].set_active(False)
    for i in range(50):
        r.run_frame()
    for i in range(40):
        net.players[i + 10].set_active(True)
        r.run_frame()
    for i in range(100):
        r.run_frame()
    for i in range(20):
        net.players[i].set_active(False)
        r.run_frame()
    for i in range(100):
        r.run_frame()
    r.prepare_stats()
    if plot:
        r.plot_stats(caption_players=False, plot_players=True, 
                     allactive=False, stat_len=stat_len, bw_height=2)
    return r


def ramp_up(player_class, do_print=False, seed=0, delayAck=False, 
         slot_per_frame=100, num_frame=100, detect_energy=True,
         stat_len=10, plot=False, min_nodes=10, max_nodes=100):
    """ ramp up from 10 nodes to 100 nodes, 1 per frame, repeat 
        this experiment 90 * 111 = 10k frames (1M time slots) 
    """
    np.random.seed(seed)
    if delayAck:
        net = Network(players=[player_class(name=str(i)) for i in range(max_nodes)], 
                       do_print=do_print, detect_energy=detect_energy)
    else:
        net = Network(players=[player_class(name=str(i)) for i in range(max_nodes)])
    r = Run(net, frame=slot_per_frame)
    for pl_idx in range(min_nodes, max_nodes):
        net.players[pl_idx].set_active(False)
    for i in range(20):
        if i % 10 == 0:
            print(".", end="")
        r.run_frame()
    for i in range(max_nodes-min_nodes):
        net.players[i + 10].set_active(True)
        r.run_frame()
        if i % 10 == 0:
            print(".", end="")
    for i in range(20):
        r.run_frame()
        if i % 10 == 0:
            print(".", end="")
    r.prepare_stats()
    if plot:
        r.plot_stats(caption_players=False, plot_players=True, 
                     allactive=False, stat_len=stat_len, bw_height=2)
    return net.history[20*slot_per_frame: (20+max_nodes-min_nodes)*slot_per_frame]


def ramp_down(player_class, do_print=False, seed=0, delayAck=True, 
         slot_per_frame=100, num_frame=100, detect_energy=True,
         stat_len=10, plot=False, min_nodes=10, max_nodes=100):
    """ ramp down from 100 nodes to 10 nodes, 1 per frame, repeat 
        this experiment 90 * 111 = 10k frames (1M time slots) in notebook 
    """
    np.random.seed(seed)
    if delayAck:
        net = Network(players=[player_class(name=str(i)) for i in range(max_nodes)], 
                       do_print=do_print, detect_energy=detect_energy)
    else:
        net = Network(players=[player_class(name=str(i)) for i in range(max_nodes)])
    r = Run(net, frame=slot_per_frame)
    for i in range(50):
        if i % 10 == 0:
            print(".", end="")
        r.run_frame()
    for i in range(max_nodes-min_nodes):
        net.players[i + 10].set_active(False)
        r.run_frame()
        if i % 10 == 0:
            print(".", end="")
    for i in range(10):
        r.run_frame()
        if i % 10 == 0:
            print(".", end="")
    r.prepare_stats()
    if plot:
        r.plot_stats(caption_players=False, plot_players=True, 
                     allactive=False, stat_len=stat_len, bw_height=2)
    return net.history[50*slot_per_frame: (50+max_nodes-min_nodes)*slot_per_frame]




def churn(player_class, num_players=100, num_steps=200,
          do_print=False, seed=None, delayAck=True, churn_rate = 1/100,
          slot_per_frame=100, detect_energy=True, stat_len=10, plot=False, 
          **kwargs):
    if delayAck:
        net = Network(players=[player_class(**kwargs) for _ in range(num_players)], 
                       do_print=do_print, detect_energy=detect_energy)
    else:
        net = Network([player_class(**kwargs) for _ in range(num_players)])

    # create schedule
    if seed:
        np.random.seed(seed)
    is_active = np.zeros((num_players, num_steps), dtype='bool')
    # starting with two nodes because with delayed ack, one node doesn't quite  work
    is_active[0, 0] = True
    is_active[-1, 0] = True
    for i in range(1,num_steps):
        is_active[:, i] = is_active[:, i-1]
        for j in range(num_players):
            if np.random.random() < churn_rate:
                is_active[j, i] = not is_active[j, i]
    r = Run(net, frame=slot_per_frame)
    for pl_idx in range(num_players):
        net.players[pl_idx].set_active(False)
    for i in range(num_steps):
        for pl_idx in range(num_players):
            net.players[pl_idx].set_active(is_active[pl_idx, i])
        r.run_frame()
    r.prepare_stats()
    if plot:
        r.plot_stats(caption_players=False, plot_players=True, 
                 allactive=False, stat_len=stat_len, bw_height=2)
    return r

