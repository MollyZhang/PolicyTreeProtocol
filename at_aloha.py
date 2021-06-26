import numpy as np
import random


class AT(object):
    """
    This is the AT-ALOHA protocol described in this paper.
    https://dl.acm.org/doi/abs/10.1145/3405671.3405817
    """

    def __init__(self, name=None, active=True,
                 t=0,
                 initial_level=1,

                 # For emptiness.
                 empty_incentive=0.1, # Probability of filling an empty slot.
                 empty_adaptation=0.99,
                 free_to_collision=1.39, # Target ratio of free to collisions.
                 min_empty_incentive = 1e-3,

                 # For kindness
                 kindness=20, # Target fraction (one every n) of slots to leave free.
                 kind_adaptation=0.98,
                 kind_incentive=0.05, # Probability of relinquishing a slot.
                 min_kind_incentive = 1e-2,

                 random_retreat=False,
                 max_num_policies = 10,
                 max_level_difference = 2,
                 start_level_offset = 3,
                 do_print=False):
        self.name = name if name else hex(random.getrandbits(16))[2:]
        self.t = t
        self.active = active

        self.kindness = kindness
        self.kind_adaptation = kind_adaptation
        self.kind_incentive = kind_incentive
        self.min_kind_incentive = min_kind_incentive

        self.empty_incentive = empty_incentive
        self.empty_adaptation = empty_adaptation
        self.free_to_collision = free_to_collision
        self.min_empty_incentive = min_empty_incentive

        self.max_num_policies = max_num_policies
        self.max_level_difference = max_level_difference
        self.random_retreat = random_retreat
        self.start_level_offset = start_level_offset
        self.do_print = do_print
        # Creates the initial policy, which consists in sending as
        # specified by the initial level.
        self.policies = [(np.random.randint(0, 2 ** initial_level), initial_level)]
        self.c_count = 0
        self.f_count = 0
        self.u_count = 0


    def get_decision(self):
        """Gets 1 if we want to send, and 0 otherwise.
        Sets:
        self.decision: decision based on the policies, to be followed if active.
        self.transmit: decision to send if we were active."""
        self.decision = 0
        self.strategy = None
        for (i, n) in self.policies:
            if self.t % (2 ** n) == i:
                self.strategy = (i, n)
                self.decision = 1
                break
        self.transmit = self.decision and self.active
        return self.transmit


    def get_bw(self):
        """Returns the bandwidth.  Uses the invariant that along every branch,
        only one node can be selected."""
        bw = 0.
        for _, n in self.policies:
            bw += 1. / (2 ** n)
        return bw


    def _print_policies(self):
        print(self.name, "policies:", [(i, 2 ** n) for i, n in self.policies])


    def set_active(self, b):
        self.active = b


    def get_depth(self):
        """Returns the depth of the tree, so we can visualize it."""
        return max([n for _, n in self.policies])

    def get_estimated_num_players(self):
        return 1. / (0.0000001 + self.empty_incentive)


    def _normalize_siblings(self, p):
        """Ensures that no two siblings are both at 1."""
        i, n = p
        if n == 0:
            return # No sibling
        m = 2 ** (n - 1)
        left_c = i % m
        right_c = m + left_c
        if (left_c, n) in self.policies and (right_c, n) in self.policies:
            # Removes siblings in favor of parent.
            self.policies.remove((left_c, n))
            self.policies.remove((right_c, n))
            self.policies.append((left_c, n - 1))
            self._normalize_siblings((left_c, n - 1))


    def _is_subpolicy(self, p1, p2):
        """Returns whether p2 is a subpolicy of p1."""
        i1, n1 = p1
        i2, n2 = p2
        return n1 < n2 and i2 % (2 ** n1) == i1


    def _clear_subtree(self, p):
        """Clears the subtree rooted at p"""
        self.policies = [pp for pp in self.policies if not self._is_subpolicy(p, pp)]


    def _normalize_tree(self, p):
        """Normalizes the policy tree for the presence of policy p"""
        self._clear_subtree(p)
        self._normalize_siblings(p)


    def _demote_node(self, i, n):
        """Demotes a node, e.g. following a collision.
        randomize: demote at random
        this: demote this
        otherwise, demotes the other slot."""
        # We demote the node and go to the unaffected child.
        assert n >= 0
        m = 2 ** n
        assert 0 <= i < m
        self.policies.remove((i, n))
        # Counts how many policies are at level n or above.  If there are
        # none, we add the demoted policy.
        if len([j for j, k in self.policies if k <= n]) == 0:
            left_i = i
            right_i = i + m
            new_i = random.choice([i, i + m])
            self.policies.append((new_i, n + 1))
        assert len(self.policies) > 0


    def _level_for_new_node(self):
        """Returns the appropriate level for a new policy node."""
        num_players = self.get_estimated_num_players()
        bw = self.get_bw()
        discrepancy = np.clip(np.log2(bw * num_players), -1, 1)
        return int(np.ceil(np.log2(num_players) + discrepancy + self.start_level_offset))


    def _insert_policy(self,):
        """Inserts a new policy at the appropriate level."""
        n = self._level_for_new_node()
        i = self.t % (2 ** n)
        new_policy = (i, n)
        self.policies.append(new_policy)
        self._normalize_tree(new_policy)


    def _simplify_tree(self):
        """Simplifies the policy tree."""
        assert len(self.policies) > 0
        for i, n in self.policies:
            assert n >= 0
            assert 0 <= i < 2 ** n
        # Chops the list, randomizing it in such a way that the preference
        # for leaves at the same level is random.
        random.shuffle(self.policies)
        self.policies.sort(key=lambda x: x[1])
        _, min_level = self.policies[0]
        self.policies = [(i, m) for (i, m) in self.policies if m < min_level + self.max_level_difference]
        self.policies = self.policies[:self.max_num_policies]


    def learn(self, collision=0, used=0, name=None):
        """collision = a collision occurred on the network;
           used = the network slot was used (by us or others)"""
        # What we did was in self.decision and self.transmit.
        # Let us try then to compute the four cases.
        if collision:
            # Collision
            self.c_count += 1
            self.kind_incentive /= self.kind_adaptation
            self.empty_incentive *= (self.empty_adaptation ** self.free_to_collision)
        elif used:
            # Used
            self.kind_incentive /= self.kind_adaptation
            self.u_count += 1
        else:
            # Free
            self.empty_incentive /= self.empty_adaptation
            self.kind_incentive *= (self.kind_adaptation ** self.kindness)
            self.f_count += 1
        self.empty_incentive = np.clip(self.empty_incentive, self.min_empty_incentive, 0.5)
        self.kind_incentive = np.clip(self.kind_incentive, self.min_kind_incentive, 0.5)
        assert self.kind_incentive > 1e-5

        if self.decision:
            # Fishes out the strategy we used to transmit.
            i, n = self.strategy
            if collision:
                # We took part in a collision.
                self._demote_node(i, n)
            else:
                # The transmission is successful.  With a certain probability,
                # we give up the other slot. This is the kindness.
                if np.random.random() < self.kind_incentive:
                    self._demote_node(i, n)

        elif not used:
            if np.random.random() < self.empty_incentive:
                self._insert_policy()

        # Finally, normalizes the policies.
        self._simplify_tree()


    def __repr__(self):
        return self.name + " " + repr(self.policies)


    def get_display_name(self):
        return "AT"


    def tick(self):
        self.t += 1


