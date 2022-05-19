# classes for blood donation graph

import numpy as np

from matching_policies import (
    random_policy,
    greedy_policy,
    rand_greedy_mix,
    nonadapt_opt,
    appx_adapt_opt,
)
from utils import get_donor_available_days


class Vertex(object):
    def __init__(self, id, data=None):
        if data is None:
            data = {}
        self.id = id  # unique vertex id
        self.edge_list = []
        self.data = data  # extra field for data

    def add_edge(self, edge):
        if edge in self.edge_list:
            raise Exception("edge already in in edge list")
        if not isinstance(edge, Edge):
            raise Exception("this is not an edge")

        self.edge_list.append(edge)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return self.id


class Edge(object):
    def __init__(self, donor, recipient, weight):
        assert isinstance(donor, Donor)
        assert isinstance(recipient, Recipient)
        assert isinstance(weight, float)
        self.donor = donor
        self.recipient = recipient
        self.weight = weight
        self.var = None

    def __eq__(self, other):
        return (self.donor.id, self.recipient.id) == (
            other.donor.id,
            other.recipient.id,
        )

    def __hash__(self):
        return hash((self.donor.id, self.recipient.id))

    def __lt__(self, other):
        return (self.donor.id, self.recipient.id) < (other.donor.id, other.recipient.id)

    def __str__(self):
        return f"Edge: {self.donor.id} -> {self.recipient.id}. weight: {self.weight}"


class Donor(Vertex):
    def __init__(self, id, data=None, last_matched_time=None):
        super().__init__(id, data)
        self.last_matched_time = last_matched_time


class Recipient(Vertex):
    def __init__(self, id, p_list, data=None):
        super().__init__(id, data)
        self.p_list = p_list  # probability that recipient is present at each time step
        self.availability_list = None
        self.fair_normalization_score = None  # this needs to be updated

    def realize_availability(self, rs_availability):
        self.availability_list = [
            rs_availability.choice([True, False], p=[p, 1.0 - p]) for p in self.p_list
        ]


class DonationGraph(object):
    def __init__(self, donor_list, recipient_list, k, num_time_steps):
        """

        :param donor_list:              list of Donor vertices
        :param recipient_list:          list of Recipient vertices
        :param k:                       min number of time steps between donor matches
        :param num_time_steps:          total number of time steps
        """
        self.donor_list = donor_list
        self.recipient_list = recipient_list
        self.k = k
        self.num_time_steps = num_time_steps
        self.edge_list = []

        assert self.k < self.num_time_steps
        assert self.k > 0
        for recipient in recipient_list:
            assert len(recipient.p_list) == num_time_steps

    def add_edge(self, donor, recipient, weight):
        assert donor in self.donor_list
        assert recipient in self.recipient_list
        edge = Edge(donor, recipient, weight)
        self.edge_list.append(edge)
        donor.add_edge(edge)
        recipient.add_edge(edge)

    def remove_disconnected_vertices(self):
        """remove all donors and recipients with no edges by overwriting"""
        self.donor_list = [d for d in self.donor_list if len(d.edge_list) > 0]
        self.recipient_list = [r for r in self.recipient_list if len(r.edge_list) > 0]

        if len(self.donor_list) == 0:
            raise Exception("all donors were removed")
        if len(self.recipient_list) == 0:
            raise Exception("all recipients were removed")

    def simulate_matching_fixedtime(
        self, policy, rs_policy, rs_recipient, donor_available_days
    ):
        """
        simulate a matching run using a policy. the policy must have the signature
            policy(donor, day, rs) --> matched_recipient
        """

        # set availability for each recipient
        for recipient in self.recipient_list:
            recipient.realize_availability(rs_recipient)

        # record matched edges for each time step
        matched_edges_by_time = [[] for _ in range(self.num_time_steps)]

        # record matched weight for each recipient
        matched_weight_by_recip = {recip.id: 0 for recip in self.recipient_list}

        for day in range(self.num_time_steps):
            for donor in filter(
                lambda d: day in donor_available_days[d.id], self.donor_list
            ):
                # match the donor via the policy
                matched_edge = policy(donor, day, rs_policy)
                if matched_edge is not None:
                    matched_edges_by_time[day].append(matched_edge)
                    matched_weight_by_recip[
                        matched_edge.recipient.id
                    ] += matched_edge.weight

        return matched_edges_by_time, matched_weight_by_recip


if __name__ == "__main__":
    """test code (not maintained)"""

    rs = np.random.RandomState(0)
    rs_policy = np.random.RandomState(0)
    rs_recipient = np.random.RandomState(1)

    # num time steps
    num_time_steps = 50
    gamma = 0.0

    # create recipients
    num_recipients = 5
    next_vertex_id = 0
    recip_list = []
    for _ in range(num_recipients):
        p_list = rs.random(num_time_steps)
        recip_list.append(Recipient(next_vertex_id, p_list))
        next_vertex_id += 1

    # create donors
    num_donors = 10
    donor_list = []
    for _ in range(num_donors):
        donor_list.append(Donor(next_vertex_id))
        next_vertex_id += 1

    k = 5
    graph = DonationGraph(donor_list, recip_list, k, num_time_steps)

    # add edges
    prob_edge = 0.4
    for donor in donor_list:
        for recip in recip_list:
            # add an edge with probability prob_edge
            if rs.rand() < prob_edge:
                weight = rs.rand()
                graph.add_edge(donor, recip, weight)

    # remove donors and recipients with no edges
    graph.remove_disconnected_vertices()

    # simulate donor availability
    seed = 0
    donor_available_days = get_donor_available_days(graph, seed)

    # matched_edges = graph.simulate_matching(random_policy, rs)
    # matched_edges = graph.simulate_matching(greedy_policy, rs)

    # lp policy.
    policy = appx_adapt_opt(graph, donor_available_days, gamma, "gurobi")
    # policy = rand_greedy_mix(0.2)

    matched_edges, recip_weight = graph.simulate_matching_fixedtime(
        policy, rs_policy, rs_recipient, donor_available_days
    )

    for i, edge_list in enumerate(matched_edges):
        print(f"day {i}:")
        for e in edge_list:
            print(str(e))
