import itertools
import logging
import sys

import numpy as np
import time
import os

try:
    import pyomo.environ as pyo
except ImportError:
    print(
        "pyomo did not import. function solve_nonadapt_opt_lp_pyomo() will not be usable."
    )

try:
    import xpress as xp
except ImportError:
    print(
        "xpress did not import. function solve_nonadapt_opt_lp_xpress() will not be usable."
    )


def get_donor_available_days(graph, seed):
    """
    simulate donor availablility (once every graph.k days)
    """

    rs = np.random.RandomState(seed)

    # randomize the last time donors were matched
    last_matched_time_list = rs.randint(-graph.k, 1, len(graph.donor_list))

    # get days each donor is available
    donor_available_days = {}
    for i, donor in enumerate(graph.donor_list):
        # find the first day when the donor is available
        # last_matched_time (int) should <= 0
        first_available_day = max(last_matched_time_list[i] + graph.k, 0)
        donor_available_days[donor.id] = list(
            range(first_available_day, graph.num_time_steps, graph.k)
        )

    return donor_available_days


def generate_filepath(output_dir, name, extension):
    # generate filepath, of the format <name>_YYYYMMDD_HHMMDD<extension>
    timestr = time.strftime("%Y%m%d_%H%M%S")
    output_string = (name + "_%s." + extension) % timestr
    return os.path.join(output_dir, output_string)


def haversine_dist(a_lat, a_lon, b_lat, b_lon):
    """
    haversine distance in km between two points (a, b) with coords (lat, lon) in decimal degrees

    should be vectorized
    """

    # earth radius (km)
    r = 6373.0

    a_lat = a_lat * np.pi / 180.0
    a_lon = np.deg2rad(a_lon)
    b_lat = np.deg2rad(b_lat)
    b_lon = np.deg2rad(b_lon)

    d = (
        np.sin((b_lat - a_lat) / 2.0) ** 2
        + np.cos(a_lat) * np.cos(b_lat) * np.sin((b_lon - a_lon) / 2.0) ** 2
    )

    return 2.0 * r * np.arcsin(np.sqrt(d))


def get_logger(logfile=None):
    format = "[%(asctime)-15s] [%(filename)s:%(funcName)s] : %(message)s"
    logger = logging.getLogger("experiment_logs")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(format)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # logging.basicConfig(filename=logfile, level=logging.DEBUG, format=format)
    else:
        logging.basicConfig(level=logging.INFO, format=format)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def solve_nonadapt_opt_lp_pyomo(graph, donor_available_days, gamma, solver_str):
    """
    solve for the optimal non-adaptive policy, where donors are notified exactly once every k days.

    return a pre-match distribution (donor_dist) for each donor, for each day they are available.

    if donor d is available at time t, donor_dist[d][t] is a list of floats where donor_dist[d][t][i] is the pre-match
    probability for d.edge_list[i]
    """

    if not "pyomo" in sys.modules:
        raise Exception("pyomo must be installed to use this function")

    model = pyo.ConcreteModel()

    # generator for all matchable edges
    def edge_t_tuples(_):
        for donor in graph.donor_list:
            # donor is available on these days
            for day in donor_available_days[donor.id]:
                for e in donor.edge_list:
                    yield e, day

    # get set of matchable edges
    model.edge_t_tuples = pyo.Set(initialize=edge_t_tuples)

    # edge variables from set
    model.edge_t_vars = pyo.Var(model.edge_t_tuples, bounds=(0.0, 1.0))

    # generator for pairs (donor_index, day) for each day donor is available
    def donor_day_generator(_):
        for donor_index, donor in enumerate(graph.donor_list):
            for day in donor_available_days[donor.id]:
                yield donor_index, day

    # return sum of all edge vars adjacent to donor on a day
    def donor_normalization(model, donor_index, day):
        # find the first day when the donor is available
        donor = graph.donor_list[donor_index]
        # return sum of donor variables
        return sum(model.edge_t_vars[(e, day)] for e in donor.edge_list) <= 1.0

    # add donor normalization constraints
    model.donor_days = pyo.Set(initialize=donor_day_generator)
    model.donor_norm_constraints = pyo.Constraint(
        model.donor_days, rule=donor_normalization
    )

    if gamma > 0.0:

        # add recipient utility variables
        def recipient_constraints(model, recipient_index):

            return (
                    sum(
                        model.edge_t_vars[(edge, t)]
                        * edge.weight
                        * graph.recipient_list[recipient_index].p_list[t]
                        for edge, t in model.edge_t_tuples
                        if edge.recipient.id == graph.recipient_list[recipient_index].id
                    )
                    * (1.0 / graph.recipient_list[recipient_index].fair_normalization_score)
                    == model.s_vars[recipient_index]
            )

        model.recipient_indices = pyo.RangeSet(0, len(graph.recipient_list) - 1)

        model.s_vars = pyo.Var(model.recipient_indices, domain=pyo.NonNegativeReals)

        # add constraints to define s_vars
        model.s_var_constraints = pyo.Constraint(
            model.recipient_indices, rule=recipient_constraints
        )

        def add_fairness_constraint(model, recipient_i, recipient_j):
            if recipient_i == recipient_j:
                return pyo.Constraint.Feasible
            else:
                return gamma * model.s_vars[recipient_i] <= model.s_vars[recipient_j]

        # add fairness constraints for all recipient pairs
        model.recip_pairs = model.recipient_indices * model.recipient_indices
        model.fairness_constraints = pyo.Constraint(
            model.recip_pairs, rule=add_fairness_constraint
        )

    # add objective
    model.obj = pyo.Objective(
        expr=sum(
            e.weight * model.edge_t_vars[(e, t)] * e.recipient.p_list[t]
            for e, t in model.edge_t_tuples
        ),
        sense=pyo.maximize,
    )

    # only xpress and gurobi have been validated
    assert solver_str in ["gurobi", "xpress"]
    solver = pyo.SolverFactory(solver_str, solver_io="python")

    solver.solve(model)

    # gather results, create a pre-match distribution for all donors on all available time steps
    # donor_dist[d][t] is the pre-match distribution for donor d at time t, indexed by edge. if None, the donor is not available
    donor_dist = {donor: [None] * graph.num_time_steps for donor in graph.donor_list}
    for donor in graph.donor_list:
        # add a donor distribution for donor_available_days
        for day in donor_available_days[donor.id]:
            donor_dist[donor][day] = [
                model.edge_t_vars[(e, day)].value for e in donor.edge_list
            ]
            ## alternate return value: dict with one key for each edge (useful for debugging)
            # donor_dist[donor][day] = {
            #     e: model.edge_t_vars[(e, day)].value for e in donor.edge_list
            # }

    return donor_dist


def solve_nonadapt_opt_lp_xpress(graph, donor_available_days, gamma):
    """
    identical to solve_nonadapt_opt_lp_pyomo, but directly uses xpress API
    """

    if not "xpress" in sys.modules:
        raise Exception("xpress must be installed to use this function")

    model = xp.problem()

    edge_t_vars = {}

    # generator for all matchable edges
    for donor in graph.donor_list:
        # donor is available on these days
        for day in donor_available_days[donor.id]:
            for e in donor.edge_list:
                edge_t_vars[(e, day)] = xp.var(lb=0.0, ub=1.0, vartype=xp.continuous)

    # add variables
    model.addVariable(list(edge_t_vars.values()))

    # add donor normalization constraints
    for donor_index, donor in enumerate(graph.donor_list):
        for day in donor_available_days[donor.id]:
            model.addConstraint(
                xp.Sum([edge_t_vars[(e, day)] for e in donor.edge_list]) <= 1.0
            )

    if gamma > 0.0:

        # create variables for recipient outcomes
        s_vars = [
            xp.var(lb=0.0, ub=xp.infinity, vartype=xp.continuous)
            for _ in range(len(graph.recipient_list))
        ]
        model.addVariable(s_vars)

        for i_recip in range(len(graph.recipient_list)):
            model.addConstraint(
                xp.Sum(
                    edge_t_vars[(edge, t)]
                    * edge.weight
                    * graph.recipient_list[i_recip].p_list[t]
                    for edge, t in edge_t_vars.keys()
                    if edge.recipient.id == graph.recipient_list[i_recip].id
                )
                * (1.0 / graph.recipient_list[i_recip].fair_normalization_score)
                == s_vars[i_recip]
            )

        # add fairness constraints for all recipient pairs
        for i_recip, j_recip in itertools.combinations(
            range(len(graph.recipient_list)), 2
        ):
            model.addConstraint(gamma * s_vars[i_recip] <= s_vars[j_recip])
            model.addConstraint(gamma * s_vars[j_recip] <= s_vars[i_recip])

    # add objective
    model.setObjective(
        xp.Sum(
            e.weight * edge_t_vars[(e, t)] * e.recipient.p_list[t]
            for e, t in edge_t_vars.keys()
        ),
        sense=xp.maximize,
    )

    model.solve()

    # gather results, create a pre-match distribution for all donors on all available time steps
    # donor_dist[d][t] is the pre-match distribution for donor d at time t, indexed by edge. if None, the donor is not available
    donor_dist = {donor: [None] * graph.num_time_steps for donor in graph.donor_list}
    for donor in graph.donor_list:
        # add a donor distribution for donor_available_days
        for day in donor_available_days[donor.id]:
            donor_dist[donor][day] = [
                model.getSolution(edge_t_vars[(e, day)]) for e in donor.edge_list
            ]

    return donor_dist
