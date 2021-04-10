#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --job-name=tts_train_De
#SBATCH --output=./train_tacotron.log
source ~/.bashrc # to make 'conda' available as a command line
source /home/hbauerec/dev/ttt/music_performer_bot/set_env_oda2.sh
python Tacotron2/train.py --n-symbols=320 --n_gpus=2 -o ./snapshots/tacotron -m Tacotron2 --epochs=500 -lr 1e-3 -bs 8 --training-files=filelists/tacotron_mel_pitch_train.txt --validation-files=filelists/tacotron_mel_pitch_valid.txt --load-mel-from-disk --resume-from-last
