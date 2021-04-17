#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=al5217 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/al5217/env/bin/:$PATH
source activate
source /vol/cuda/11.1.0-cudnn8.0.4.30/setup.sh
TERM=vt100 # or TERM=xterm
python3 -u -W ignore train_models_normal.py --train_all --epochs 20 --batch_size 128 --num_models 30 --learning_rate 0.00025
/usr/bin/nvidia-smi
uptime