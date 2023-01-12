#!/bin/bash 

#SBATCH -p compute
#SBATCH -t 168:00:00
#SBATCH -J unbiased_bulk_analysis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28              # total number of mpi tasks requested
#SBATCH --mail-user=bdallin@wisc.edu
#SBATCH --mail-type=all  # email me when the job starts

# INSTRUCTIONS:
# The -t command specifies how long this simulation will run for (in HOURS:MINUTES:SECONDS). Try to estimate this as best you can
# as the simulation will temrinate after this time.
# The -J flag species the name of the simulation
# The --mail-user command will send you email when the job runs / terminates
# The --nodes flag designates how many nodes are reserved. On swarm this should usually be 1, since we do not have efficient internode communication.
# If you do try multiple nodes (which will have significant overhead), make sure to use the mpi and not thread-mpi version of gromacs.
# The --ntasks-per-node flag specifies how many cores on each node will be used. This should match however many threads you use in the commands below.
# Note that there are 28 logical cores total on each node; its recommended to not use more than 1 core per 500 atoms. 

## MAKE FUNCTIONS AND ALIASES AVAIABLE
source ~/.bashrc

## ACTIVATE SAM ANALYSIS CONDA ENVIRONMENT
workon sam_analysis

## MOVE TO DRIVER DIRECTORY
cd ./sam_analysis/drivers

## SET PYTHON FILE NAME
py_file="unbiased_analysis.py"

## RUN THE ANALYSIS FILE
python "$py_file" > slurm-${SLURM_JOB_ID}.out 2>&1

