#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load cuda
./hw4 5120 5000 /scratch/$USER/output.5120.5000.gpu
./hw4 5120 5000 /scratch/$USER/output.5120.5000.gpu
./hw4 5120 5000 /scratch/$USER/output.5120.5000.gpu
