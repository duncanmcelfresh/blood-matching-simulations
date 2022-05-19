# use geographic coordinate points to prepare donor/recip files
import argparse

import pandas as pd
import numpy as np
import os
import time

from utils import generate_filepath


def create_donation_files(
    points_file, output_dir, output_name, num_donors, num_recipients, seed=0
):
    """read a file of geographic coordinates and write a file of donors and recipients"""

    rs = np.random.RandomState(seed)

    points_df = pd.read_csv(points_file, sep=" ", skiprows=1)

    # select random donors and recipients
    keep_inds = rs.choice(len(points_df), (num_donors + num_recipients), replace=False)
    donor_points = points_df.iloc[keep_inds[:num_donors]]
    recipient_points = points_df.iloc[keep_inds[num_donors:]]

    # write donor and recip files
    assert os.path.exists(output_dir)

    donor_filepath = os.path.join(
        output_dir, generate_filepath(output_dir, output_name + "_donors", "csv")
    )
    recipient_filepath = os.path.join(
        output_dir, generate_filepath(output_dir, output_name + "_recipients", "csv")
    )
    print(f"writing donor points to file: {donor_filepath}")
    with open(donor_filepath, "w") as f:
        # write header
        f.write("lat lon\n")
        for i, row in donor_points.iterrows():
            f.write("{lat:.5f} {lon:.5f}\n".format(lat=row["lat"], lon=row["lon"]))

    print(f"writing recipient points to file: {recipient_filepath}")
    with open(recipient_filepath, "w") as f:
        # write header
        f.write("lat lon\n")
        for i, row in recipient_points.iterrows():
            f.write("{lat:.5f} {lon:.5f}\n".format(lat=row["lat"], lon=row["lon"]))


if __name__ == "__main__":

    # parse some args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--points-file", type=str, help="file with random lat/lon points", default="",
    )
    parser.add_argument(
        "--output-dir", type=str, help="directory for writing output", default=""
    )
    parser.add_argument(
        "--output-name", type=str, help="name ofr output file", default=""
    )
    parser.add_argument(
        "--num-donors",
        type=int,
        help="number of donors to draw from random points",
        default=1,
    )
    parser.add_argument(
        "--num-recipients",
        type=int,
        help="number of recipients to draw from random points",
        default=1,
    )
    parser.add_argument(
        "--seed", type=int, help="random seed for generating points", default=0,
    )

    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="if set, use a fixed arg string for debugging. otherwise, parse args.",
        default=False,
    )
    args = parser.parse_args()

    if args.DEBUG:
        arg_str = "<example arg string>"
        args = parser.parse_args(arg_str.split())

    create_donation_files(
        args.points_file,
        args.output_dir,
        args.output_name,
        args.num_donors,
        args.num_recipients,
        seed=args.seed,
    )
