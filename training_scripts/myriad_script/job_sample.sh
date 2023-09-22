#!/bin/bash -l

# Batch script to run an OpenMP threaded job under SGE.

# Request wallclock time (format hours:minutes:seconds).
#$ -l h_rt=40:0:0

# Request N gigabyte of RAM for each core/thread 
# (must be an integer followed by M, G, or T)
#$ -l mem=192G

# Request 15 gigabyte of TMPDIR space (default is 10 GB - remove if cluster is diskless)
#$ -l tmpfs=15G

# Set the name of the job.
#$ -N train_1e_100_5

# Request for cores.
#$ -pe smp 36

# Request 1 GPU
#$ -l gpu=1

# Set the working directory to somewhere in your scratch space.
# Replace "<your_UCL_id>" with your UCL user ID
#$ -wd /home/<your_UCL_id>/codes/rl_waypoint_mrta

# Load python
module load python/3.9.10

# 8. Run the application.
source training_scripts/train.bash
