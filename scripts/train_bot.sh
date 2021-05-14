#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:1
#SBATCH --job-name=transformer_lm
#SBATCH --output=./train_bot.log
source ~/.bashrc # to make 'conda' available as a command line
source /home/hbauerec/dev/ttt/music_performer_bot/set_env_oda2.sh
python performer_bot/run_multitask.py --path data/bot_data/ --epochs 50
