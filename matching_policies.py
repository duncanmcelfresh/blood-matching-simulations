# matching policy functions
#
# each policy function has the signature:
#       policy(donor, day, rs) --> matched_edge
import numpy as np

from utils import solve_nonadapt_opt_lp_pyomo, solve_nonadapt_opt_lp_xpress


def random_policy(donor, day, rs):
    """return a random edge to an available compatible donor"""

    # get available compatible recipients
    matchable_edges = list(
        filter(lambda e: e.recipient.availability_list[day], donor.edge_list)
    )
    if matchable_edges == []:
        return None
    else:
        return rs.choice(matchable_edges)


def greedy_policy(donor, day, rs):
    """return an available edge with maximal edge weight (break ties randomly)"""

    # get available compatible recipients
    matchable_edges = list(
        filter(lambda e: e.recipient.availability_list[day], donor.edge_list)
    )
    if matchable_edges == []:
        return None
    else:
        # find the maximal edge weight
        max_wt = max(e.weight for e in matchable_edges)
        return rs.choice(list(filter(lambda e: e.weight == max_wt, matchable_edges)))


def rand_greedy_mix(p_greedy):
    """
    return a policy (function handle) that selects greedy with probability p_greedy, and random otherwise
    """

    def policy(donor, day, rs):
        if rs.rand() < p_greedy:
            return greedy_policy(donor, day, rs)
        else:
            return random_policy(donor, day, rs)

    return policy


def nonadapt_opt(graph, donor_available_days, gamma, solver_str):
    """
    return the optimal non-adaptive policy, using the solution to the optimal-non-adaptive policy LP
    """

    # get an optimal non-adaptive pre-match distribution
    assert solver_str in ["gurobi", "xpress"]

    if solver_str == "gurobi":
        donor_dist = solve_nonadapt_opt_lp_pyomo(
            graph, donor_available_days, gamma, "gurobi"
        )
    if solver_str == "xpress":
        donor_dist = solve_nonadapt_opt_lp_xpress(graph, donor_available_days, gamma)

    def policy(donor, day, rs):
        # draw an edge according to the donor distribution

        # if the sum isn't 1, add a None option
        if sum(donor_dist[donor][day]) < 1.0:
            pre_matched_edge = rs.choice(
                donor.edge_list + [None],
                p=donor_dist[donor][day] + [1.0 - sum(donor_dist[donor][day])],
            )
            if pre_matched_edge is None:
                return None
        else:
            pre_matched_edge = rs.choice(donor.edge_list, p=donor_dist[donor][day])

        if pre_matched_edge.recipient.availability_list[day]:
            return pre_matched_edge
        else:
            return None

    return policy


def appx_adapt_opt(graph, donor_available_days, gamma, solver_str):
    """
    return an approximate adaptive policy, based on the optimal non-adaptive policy, using the solution to the
    optimal-non-adaptive policy LP
    """

    # get an optimal non-adaptive pre-match distribution
    assert solver_str in ["gurobi", "xpress"]

    if solver_str == "gurobi":
        donor_dist = solve_nonadapt_opt_lp_pyomo(
            graph, donor_available_days, gamma, "gurobi"
        )
    if solver_str == "xpress":
        donor_dist = solve_nonadapt_opt_lp_xpress(graph, donor_available_days, gamma)

    def policy(donor, day, rs):
        # draw an available edge according to the donor distribution

        # if no available edges, don't match
        matchable_edges = list(
            filter(lambda e: e.recipient.availability_list[day], donor.edge_list)
        )
        if matchable_edges == []:
            # if no available edges, don't match
            return None

        else:
            # try to pre-select an edge
            # (copy/paste of the non-adaptive policy)
            # if the sum isn't 1, add a None option
            if sum(donor_dist[donor][day]) < 1.0:
                pre_matched_edge = rs.choice(
                    donor.edge_list + [None],
                    p=donor_dist[donor][day] + [1.0 - sum(donor_dist[donor][day])],
                )
            else:
                pre_matched_edge = rs.choice(donor.edge_list, p=donor_dist[donor][day])

            # return pre-matched edge if it exists, and if it's available
            if pre_matched_edge is not None:
                if pre_matched_edge.recipient.availability_list[day]:
                    return pre_matched_edge

            # if no pre-matched edge, or pre-matched edge is not available, use the random-greedy mix
            # (copy-paste from rand_greedy_mix)
            if rs.rand() < (1.0 - gamma):
                return greedy_policy(donor, day, rs)
            else:
                return random_policy(donor, day, rs)

    return policy
