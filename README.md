# `blood-matching`: Matching Algorithms for Blood Donation

### Dependencies
- either [gurobi](https://www.gurobi.com/) & gurobipy, or [xpress](https://www.fico.com/en/products/fico-xpress-solver) & xpress python API
- pyomo >=5.1. **Note:** If using xpress, then pyomo version 5.7.1 is required
- numpy
- pandas

### Software Versions
Our experiments were run in Python 3.6.10, using a conda environment with the following package versions:
- gurobipy (gurobi) 9.0.3 
- pyomo 5.7.1
- numpy 1.19.5
- pandas 1.1.3

Installing all required packages should take approximately 10 minutes on a personal computer.


### Files & Usage

`donationgraph.py`: core classes for implementing matching.
`
matching_policies.py`: functions & function generators for matching policies.

`utils.py`: various helper functions.

`experiment.py`: code for running a matching experiment.

#### Geodata

Files in this directoy are used to generate geographic coordinates randomly, according to population distributions.

`geodata/generate_points.py`: generate random coordinates using [GPW data](http://sedac.ciesin.columbia.edu/data/collection/gpw-v4/documentation). This requires GPW data to be available locally.

`geodata/prepare_donation_files.py`: read a list of geographic points and randomly generate a list of donors and recipients by drawing from these points.


# Reproducing Experimental results

To reproduce the experiments described in our paper, follow the steps below.

These instructions are for running our experiments on a single lat/lon bounding box (e.g., a city).

**Runtime:** Experiments for each "city" of 500 donors and 10 recipients should take at most one hour to complete on a personal computer. All experiments were run on a dual-core laptop with 8GB memory. 

**City Bounding Boxes:** The four bounding boxes used to generate data for our experiments are:
- Istanbul: min lat.: 40.788864; max lat.: 41.139918; min lon.: 28.549032; max lon. : 29.349461
- Jakarta: min lat.: -6.36892; max lat.: -6.069024; min lon.: 106.647546; max lon.: 106.990576 
- San Francisco: min lat.: 37.703881; max lat.: 37.81468; min lon.: -122.542452; max lon.: -122.352124 
- Sao Paulo: min lat.: -23.796439; max lat.: -23.374034; min lon.: -46.856456; max lon.: -46.285069 

For each city, 10,000 datapoints were generated using a random seed of 0 (instructions below).

### 1. Generate random geographical data

1a) First, download the GPW data, [here](https://sedac.ciesin.columbia.edu/data/collection/gpw-v4/documentation). You will need to read the documentation provided on the SEDAC website for instructions on downloading the data. You will need to have the 30 arc-second datasets in an accessible directory.

1b) Run the script `geodata/generate_points.py`, with command line arguments specifying the lat/lon bounding box of interest, the number of points (we used 10000), the location of the GWP data, and the output name and directory. For example:

```commandline
python -m geodata.generate_points 
     --gpw-dir /data/gpw-v4-population-density-rev11_2020_30_sec_asc \
     --output-dir /data/random-points \
     --output-name city1 \
     --num-points 10000 \
     --lat-bounds -6.368920 -6.069024 \
     --lon-bounds 106.647546 106.990576 \
```

### 2. Generate random donors and recipients

Using the random points generated during step 1, create a file with donors and recipients. Do this with script `geodata/prepare_donation_files.py`. For example:

```commandline
python -m geodata.prepare_donation_files 
    --points-file /data/random-points/city1.csv \
    --output-dir /data/donors-and-recipients \
    --output-name city1 \
    --num-donors 500 \
    --num-recipients 10 \
```

### 3. Run experiments

The script `experiemnt.py` will run each matching policy on a donation graph based on the donor and recipient files. For example:

```commandline
python -m experiment 
    --donor-file /data/donors-and-recipients/city1_donors.csv \
    --recipient-file /data/donors-and-recipients/city1_recipients.csv \
    --name city1 \
    --gamma-list 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --frac-static-recip 0.5 \
    --num-days 30 \
    --num-random-trials 100 \
    --num-realizations 50 \
    --k 7 \
    --distance-thresh 15 \
    --solver gurobi \
```

The arguments specified above are the same used in our experiments.


### 4. Generate Plots

All plots can be generated using the script `plots.py`. Before running this script, make sure to change the file paths of the result CSVs generated during the previous steps, and the file path where the resulting image will be written.
