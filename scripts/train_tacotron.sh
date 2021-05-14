#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --mem=120G
###BATCH --partition=scavenge
#SBATCH --gres=gpu:4
#SBATCH --job-name=tts_train_De
#SBATCH --output=./train_tacotron.log
source ~/.bashrc # to make 'conda' available as a command line
source /home/hbauerec/dev/ttt/music_performer_bot/set_env_oda2.sh

python Tacotron2/train.py -o ./snapshots/tacotron -m Tacotron2 --epochs=1000 -lr 1e-3 -bs 8 --training-files=filelists/tacotron_mel_pitch_train.txt --validation-files=filelists/tacotron_mel_pitch_valid.txt --load-mel-from-disk --resume-from-last

mv ./train_tacotron.log log/train_tacotron_$(date  '+%Y%m%di_%H%M%S').log
