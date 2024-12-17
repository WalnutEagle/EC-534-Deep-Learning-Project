#!/bin/bash -l

#$ -P ec523kb                # Specify the SCC project name
#$ -l h_rt=30:00:00          # Specify the hard time limit of 24 hours
#$ -N CMPLX                  # Specify the job name
#$ -l mem_per_core=6G        # Request memory per core (adjust as needed)
#$ -pe omp 14                # Request 14 cores for the job
#$ -l gpus=2                 # Request 2 GPUs for the job
#$ -l gpu_type=A100          # Specify the GPU type as A100
#$ -j n                      # Merge error and output streams into a single file
#$ -o job_output_CMPLX.log   # Specify the output file for stdout
#$ -e job_error_CMPLX.log    # Specify the output file for stderr
#$ -m beas                   # Send email when the job begins (b) and ends (e)
#$ -M adk1361@bu.edu         # Replace with your email address

# Load the necessary modules
module load cuda/11.2 python3/3.8.3 pytorch cudnn

# Set the working directory
cd /projectnb/ec523kb/projects/teams_Fall_2024/Team_11/Adwait

# Run the Jupyter notebook using nbconvert to execute it
papermill /projectnb/ec523kb/projects/teams_Fall_2024/Team_11/Adwait/Work_on_this_code/Phew/CGANMRCMPLX.ipynb
