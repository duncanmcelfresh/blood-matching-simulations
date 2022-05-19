# run a matching experiment with donor/recipient locations
import argparse
import os

import numpy as np
import pandas as pd

from donationgraph import Donor, Recipient, DonationGraph
from matching_policies import (
    random_policy,
    greedy_policy,
    rand_greedy_mix,
    nonadapt_opt,
    appx_adapt_opt,
)
from utils import (
    get_donor_available_days,
    generate_filepath,
    get_logger,
    haversine_dist,
)

P_VALUES = [0.1, 0.9]
MEAN_DURATION = 4
DELIMITER = ";"


def experiment(args):
    """
    generate a random donation graph using donor/recipient location (lat/lon in decimal degrees).

    randomly assign recipients to be static (always available) or dynamic (sometimes available).
    for dynamic recipients, randomly generate availability prob (p_list).

    pre-procesesing of donors and recipients is as follows:
    - for each donor, randomly assign a ``nominal'' edge weight w_0 on U[0.01, 0.05], and a decay parameter k chosen
        from [10, 20]. edge weight for the donor is w_0 * exp(-d / k) where d is the distance between donor/recipient
    - for each recipient, find donors within args.distance_thresh km of each other, and add edges
    - recipient availability is either high (p=0.9) or low (p=0.1) for a sequence of k days (alternating).
        with sequence length 1 + Poisson(3) -- so mean duration 4 days
    """

    # -----------
    # bookkeeping
    # -----------

    rs = np.random.RandomState(args.seed)

    # output file - write header
    assert os.path.exists(args.output_dir)
    output_file = generate_filepath(
        args.output_dir, f"blood_experiment_{args.name}", "csv"
    )
    log_file = generate_filepath(
        args.output_dir, f"blood_experiment_{args.name}_LOGS", "txt"
    )
    logger = get_logger(logfile=log_file)
    logger.info("writing to output file: {}".format(output_file))

    # columns for results file
    col_list = [
        "realization_num",
        "method",
        "total_matched_weight",
        "num_matched_edges",
        "recipient_matched_weight",
    ]

    # write the file header: write experiment parameters to the first line of the output file
    with open(output_file, "w") as f:
        f.write(str(args) + "\n")
        f.write((DELIMITER.join(len(col_list) * ["%s"]) + "\n") % tuple(col_list))

    # ---------------------------------
    # read input, create donation graph
    # ---------------------------------

    # read donors & recipients
    df_donor = pd.read_csv(args.donor_file, sep=" ")
    df_recipient = pd.read_csv(args.recipient_file, sep=" ")

    if args.limit_recips >= 0:
        assert args.limit_recips < len(df_recipient)
        df_recipient = df_recipient.iloc[
            rs.choice(len(df_recipient), args.limit_recips, replace=False), :
        ]

    if args.limit_donors >= 0:
        assert args.limit_donors < len(df_donor)
        df_donor = df_donor.iloc[
            rs.choice(len(df_donor), args.limit_donors, replace=False), :
        ]

    # add id column
    df_donor["id"] = list(range(len(df_donor)))
    df_recipient["id"] = list(range(len(df_donor), len(df_donor) + len(df_recipient)))

    # assign donor weight parameters
    min_w0 = 0.01
    max_w0 = 0.08
    # df_donor["w0"] = rs.random(len(df_donor)) * (max_w0 - min_w0) + min_w0
    df_recipient["w0"] = rs.random(len(df_recipient)) * (max_w0 - min_w0) + min_w0

    k_values = [5, 10, 20]
    df_donor["k"] = rs.choice(k_values, len(df_donor), replace=True)

    # create donors and recipients
    donor_dict = dict()
    for id in df_donor["id"]:
        donor_dict[id] = Donor(id)

    recip_dict = dict()
    for id in df_recipient["id"]:
        # randomly assign to static or dynamic
        if rs.random() < args.frac_static_recip:
            # static recipient
            p_list = np.ones(args.num_days)
            data = {"static": True}
        else:
            # dynamic recipient
            p_list = get_p_list(args.num_days, rs)
            data = {"static": False}

        recip_dict[id] = Recipient(id, p_list, data=data)

    graph = DonationGraph(
        list(donor_dict.values()), list(recip_dict.values()), args.k, args.num_days,
    )

    # for each recipient, find compatible donors and add weights
    for i, row in df_recipient.iterrows():
        # add a temporary col for donor distance
        recip_w0 = row["w0"]

        df_donor["weight"] = np.zeros(len(df_donor))
        df_donor["dist"] = haversine_dist(
            row["lat"], row["lon"], df_donor["lat"], df_donor["lon"]
        )

        # calculate edge weights for only compatible donors
        compat_donors = df_donor["dist"] <= args.distance_thresh
        df_donor.loc[compat_donors, "weight"] = edge_weight(
            recip_w0,  # df_donor[compat_donors]["w0"],
            df_donor[compat_donors]["k"],
            df_donor[compat_donors]["dist"],
        )

        # get df slice of only compatible donors
        df_compat_donors = df_donor.loc[df_donor["dist"] <= args.distance_thresh]

        # add edges for each compatible donor
        for i_donor_row, donor_row in df_compat_donors.iterrows():
            graph.add_edge(
                donor_dict[donor_row["id"]], recip_dict[row["id"]], donor_row["weight"]
            )

    # remove donors and recipients with no edges
    graph.remove_disconnected_vertices()

    # simulate donor availability
    donor_available_days = get_donor_available_days(graph, args.seed)

    # calculate "fair" weight for each recipient
    # do this by simulating several random trajectories
    recip_outcomes = {id: 0 for id in recip_dict.keys()}
    total_matched_edges = 0
    for i_trial in range(args.num_random_trials):
        if (i_trial % 10) == 0:
            logger.info(f"random trial {i_trial}  {args.num_random_trials}...")
        matched_edges, recip_weight = graph.simulate_matching_fixedtime(
            random_policy, rs, rs, donor_available_days
        )
        for recip_id, weight in recip_weight.items():
            recip_outcomes[recip_id] += weight
        total_matched_edges += sum(len(e_list) for e_list in matched_edges)

    mean_recip_outcomes = {
        id: weight / float(args.num_random_trials)
        for id, weight in recip_outcomes.items()
    }
    mean_matched_weight = sum(mean_recip_outcomes.values())
    mean_matched_edges = total_matched_edges / float(args.num_random_trials)

    # add fair weights to recipients
    for recip in graph.recipient_list:
        recip.fair_normalization_score = mean_recip_outcomes[recip.id]

    # write these aggregated outcomes as a policy
    with open(output_file, "a") as f:
        f.write(
            (DELIMITER.join(len(col_list) * ["{}"]) + "\n").format(
                None,
                "aggregated_random",
                mean_matched_weight,
                mean_matched_edges,
                mean_recip_outcomes,
            )
        )

    # function for aggregating output
    def write_results(realization_num, method_name, matched_edges, recip_weight):
        """gather relevant outputs and write them to file

        recipient_fair_weight_dict[i] is equal to the fair weight for recipient with id i"""

        num_matched_edges = sum(len(e_list) for e_list in matched_edges)

        total_matched_weight = sum(recip_weight.values())

        # temporary sanity check
        total_matched_weight_check = sum(
            e.weight for e_list in matched_edges for e in e_list
        )
        assert np.isclose(total_matched_weight, total_matched_weight_check)

        with open(output_file, "a") as f:
            f.write(
                (DELIMITER.join(len(col_list) * ["{}"]) + "\n").format(
                    realization_num,
                    method_name,
                    total_matched_weight,
                    num_matched_edges,
                    recip_weight,
                )
            )

    # -----------------
    # simulate matching
    # -----------------

    # dict with each policy that runs for multiple realizations
    policy_dict = {
        "mix_05": rand_greedy_mix(0.5),
    }

    # add LP-based policies to the dict
    for gamma in args.gamma_list:
        logger.info(f"solving LP for gamma={gamma} (non-adapt)...")
        policy_dict[f"lp_non_adapt_{gamma}"] = nonadapt_opt(
            graph, donor_available_days, gamma, args.solver
        )
        logger.info(f"solving LP for gamma={gamma} (appx-adapt)...")
        policy_dict[f"lp_appx_adapt_{gamma}"] = appx_adapt_opt(
            graph, donor_available_days, gamma, args.solver
        )

    # initialize donor availability
    rs_recipient = np.random.RandomState(args.seed)

    # run greedy (only one realization)
    rs_policy = np.random.RandomState(args.seed)
    matched_edges, recip_weight = graph.simulate_matching_fixedtime(
        greedy_policy, rs_policy, rs_recipient, donor_available_days
    )
    write_results(None, "greedy", matched_edges, recip_weight)

    for realization_num in range(args.num_realizations):
        logger.info(
            f"running realization number {realization_num + 1} (of {args.num_realizations})"
        )
        for policy_name, policy_func in policy_dict.items():
            logger.info(f"running policy {policy_name}...")

            # re-initialize donor availability (same) and policy (different)
            rs_recipient = np.random.RandomState(args.seed)
            rs_policy = np.random.RandomState(args.seed + realization_num)
            matched_edges, recip_weight = graph.simulate_matching_fixedtime(
                policy_func, rs_policy, rs_recipient, donor_available_days
            )

            # aggregate and write results
            write_results(realization_num, policy_name, matched_edges, recip_weight)

    logger.info("done")


def get_p_list(num_days, rs):
    """return a list of p values, which are in an alternating sequence of p values (large or small), where
    each sequence has random length on [1, \infty) (mean 4)"""

    assert num_days > 0

    start_ind = rs.choice([0, 1])
    p_list = []

    for i in range(num_days):
        # add another sequence
        remaining_days = num_days - len(p_list)
        p_value = P_VALUES[(start_ind + i) % 2]
        seq_length = min(1 + rs.poisson(3), remaining_days)
        p_list.extend([p_value] * seq_length)

        if len(p_list) == num_days:
            break

    assert len(p_list) == num_days
    return p_list


def edge_weight(w0, k, dist):
    """
    edge weight: w0 * exp(-dist/k)

    should be vectorized
    """
    return w0 * np.exp(-dist / k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed", type=int, help="random seed", default=0,
    )
    parser.add_argument("--output-dir", type=str, help="output directory")
    parser.add_argument(
        "--donor-file", type=str, help="file containing donor locations"
    )
    parser.add_argument(
        "--recipient-file", type=str, help="file containing recipient locations"
    )
    parser.add_argument("--name", type=str, help="name of experiment")
    parser.add_argument(
        "--frac-static-recip",
        type=float,
        help="fraction of recipients to make static",
        default=0.5,
    )
    parser.add_argument(
        "--gamma-list",
        type=float,
        nargs="+",
        help="fairness parameter for LP-based solutions",
    )
    parser.add_argument(
        "--num-random-trials",
        type=int,
        default=50,
        help="number of random trials for estimating fair weights",
    )
    parser.add_argument(
        "--num-realizations",
        type=int,
        default=50,
        help="number of demand realizations",
    )
    parser.add_argument(
        "--num-days", type=int, help="fraction of recipients to make static", default=10
    )
    parser.add_argument("--k", type=int, help="donor matching rate limit", default=5)
    parser.add_argument(
        "--distance-thresh",
        type=float,
        help="maximum distance between compatible donor/recip pair (km)",
        default=15,
    )
    parser.add_argument(
        "--limit-recips",
        type=int,
        help="if set, randomly drop recipients until there are this many recipients left",
        default=-1,
    )
    parser.add_argument(
        "--limit-donors",
        type=int,
        help="if set, randomly drop donors until there are this many donors left",
        default=-1,
    )
    parser.add_argument(
        "--solver",
        type=str,
        choices=["gurobi", "xpress"],
        help="LP solver (only gurobi or xpress are implemented)",
    )

    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="if set, use a fixed arg string. otherwise, parse args.",
        default=False,
    )

    args = parser.parse_args()

    if args.DEBUG:
        arg_str = (
            "<example arg string>"
        )

        args_fixed = parser.parse_args(arg_str.split())
        experiment(args_fixed)
    else:
        args = parser.parse_args()
        experiment(args)
