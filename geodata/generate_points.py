# read GPW data and generate points randomly within a lat/lon rectangle bounding box
# intended for 30 arc-sec GPW v4 Rev11. see: http://sedac.ciesin.columbia.edu/data/collection/gpw-v4/documentation
import argparse
import glob
import os
import time
import numpy as np

from utils import generate_filepath

NODATA = -9999
GRID_SPACING = 90.0 / 10800.0


def gen_random_population(
    gpw_dir,
    output_dir,
    output_name,
    num_points,
    min_lat,
    max_lat,
    min_lon,
    max_lon,
    seed=0,
):
    """
    generate random points according to GPW population density

    output_dir : directory for output file
    num_points : number of random points to generate
    gpw_dir  : directory of GPW data files (ASCII format, 30 arc-sec)
    min_lat, max_lat : min / max latitute of bounding box in decimal degrees
    min_lon, max_lon : min / max longitude of bounding box in decimal degrees
    seed : random seed for drawing coords
    """

    data, new_max_lat, new_min_lon = read_gpw_data(
        gpw_dir, min_lat, max_lat, min_lon, max_lon
    )

    # generate 1d coordinate list to represent distribution - only grids with nonzero density
    y_coord, x_coord = data.nonzero()
    coord_list = list(zip(y_coord, x_coord))

    pop_list = np.array([data[c] for c in coord_list])

    # normalize
    pop_list = pop_list / pop_list.sum()

    # draw num_points indices from pop_list
    rs = np.random.RandomState(seed)
    point_inds = rs.choice(len(coord_list), num_points, p=pop_list, replace=True)

    # generate lat/lon coords uniformly within each grid square by adding uniform random noise on [0, 1]^2
    noise = rs.random(size=(num_points, 2))

    # final coordinates with added noise (without offset)
    pop_coords = (
        np.array([coord_list[ind] + noise[i] for i, ind in enumerate(point_inds)])
        * GRID_SPACING
    )

    # adjust the new lat/lon
    final_lat = new_max_lat - pop_coords[:, 0]
    final_lon = new_min_lon + pop_coords[:, 1]

    # write points to file
    assert os.path.exists(output_dir)

    new_filepath = os.path.join(
        output_dir, generate_filepath(output_dir, output_name, "csv")
    )
    print(f"writing points to file: {new_filepath}")
    with open(new_filepath, "w") as f:
        # write header
        f.write(
            f"min_lat={min_lat} max_lat={max_lat} min_lon={min_lon} max_lon={max_lon} seed={seed} num_points={num_points}\n"
        )
        f.write("lat lon\n")
        for i in range(num_points):
            f.write("{lat:.5f} {lon:.5f}\n".format(lat=final_lat[i], lon=final_lon[i]))


def read_gpw_data(gpw_dir, min_lat, max_lat, min_lon, max_lon):
    """
    read a subset of data from  GPW 30 arc-sec dataset.

    bounding box must be in one 'section' of GPW data (8 sections total, 90 deg square

    gpw_dir  : directory of GPW data files (ASCII format, 30 arc-sec)
    min_lat, max_lat : min / max latitute of bounding box in decimal degrees
    min_lon, max_lon : min / max longitude of bounding box in decimal degrees
    """

    assert min_lat < max_lat
    assert min_lon < max_lon

    # determine which section the ll of this bounding box is in
    ll_sec = get_grid_section(min_lat, min_lon)
    ur_sec = get_grid_section(max_lat, max_lon)

    # both ll and ur points must be in same grid
    if ll_sec[0] != ur_sec[0]:
        raise NotImplemented

    # grab only the necessary data
    gpw_file = glob.glob(
        os.path.join(gpw_dir, f"gpw_*_population_*_sec_{ll_sec[0]}.asc")
    )
    assert len(gpw_file) == 1
    gpw_file = gpw_file[0]

    print(f"reading GPW data for section {ll_sec[0]}...")
    data = np.loadtxt(gpw_file, skiprows=6)

    # the 30 arc-sec data should be a 10800 x 10800 grid
    assert data.shape == (10800, 10800)

    # convert the lat/lon of the bounding box into grid coordinates (inclusive)
    min_y = int(
        np.floor((ll_sec[1] - max_lat) / GRID_SPACING)
    )  # distance in grid points from the ul corner of the section
    max_y = int(np.ceil((ll_sec[1] - min_lat) / GRID_SPACING))
    min_x = int(np.floor((min_lon - ll_sec[2]) / GRID_SPACING))
    max_x = int(np.ceil((max_lon - ll_sec[2]) / GRID_SPACING))

    # read only relevant data
    data_bb = data[min_y:max_y, min_x:max_x]

    # replace nodata values with 0
    data_bb[data_bb == NODATA] = 0.0

    # for fun, plot the data using matplotlib : )
    # im = plt.imshow(data_bb)

    # get new ul coords
    new_max_lat = ll_sec[1] - min_y * GRID_SPACING
    new_min_lon = ll_sec[2] + min_x * GRID_SPACING

    # return the relevant data grid and ul of the bounding box (lat, lon)
    return data_bb, new_max_lat, new_min_lon


def get_grid_section(lat, lon):
    """return the index of the grid section (1-8) of the lat-lon point in decimal degrees,
    and return the ul (lat, lon) of the section"""
    assert (lat <= 90) and (lat >= -90)
    assert (lon <= 180) and (lon >= -180)

    if lat >= 0:
        if (lon >= -180) and (lon < -90):
            return 1, 90.0, -180.0
        if (lon >= -90) and (lon < -0):
            return 2, 90.0, -90.0
        if (lon >= -0) and (lon < 90):
            return 3, 90.0, -0.0
        if (lon >= 90) and (lon <= 180):
            return 4, 90.0, 90.0
    if lat < 0:
        if (lon >= -180) and (lon < -90):
            return 5, 0.0, -180.0
        if (lon >= -90) and (lon < -0):
            return 6, 0.0, -90.0
        if (lon >= -0) and (lon < 90):
            return 7, 0.0, -0.0
        if (lon >= 90) and (lon <= 180):
            return 8, 0.0, 90.0


if __name__ == "__main__":

    # parse some args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpw-dir",
        type=str,
        help="directory with GPW population data files",
        default="",
    )
    parser.add_argument(
        "--output-dir", type=str, help="directory for writing output", default=""
    )
    parser.add_argument(
        "--output-name", type=str, help="name ofr output file", default=""
    )
    parser.add_argument(
        "--num-points",
        type=int,
        help="number of points to draw from population distribution",
        default=8,
    )
    parser.add_argument(
        "--seed", type=int, help="random seed for generating points", default=0,
    )
    parser.add_argument(
        "--lat-bounds",
        type=float,
        nargs=2,
        help="min and max latitude for bounding box(decimal degrees)",
    )
    parser.add_argument(
        "--lon-bounds",
        type=float,
        nargs=2,
        help="min and max longitude for bounding box(decimal degrees)",
    )
    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="if set, use a fixed arg string for debugging. otherwise, parse args.",
        default=False,
    )
    args = parser.parse_args()

    if args.DEBUG:
        arg_str = "<example-arg-string>"
        args = parser.parse_args(arg_str.split())

    gen_random_population(
        args.gpw_dir,
        args.output_dir,
        args.output_name,
        args.num_points,
        args.lat_bounds[0],
        args.lat_bounds[1],
        args.lon_bounds[0],
        args.lon_bounds[1],
        seed=args.seed,
    )
