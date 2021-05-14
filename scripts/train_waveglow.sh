#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:4
#SBATCH --job-name=train_transformer_de
#SBATCH --output=./train_waveglow.log
source ~/.bashrc # to make 'conda' available as a command line
source /home/hbauerec/dev/ttt/music_performer_bot/set_env_oda2.sh
python Tacotron2/train.py -m WaveGlow -o snapshots/waveglow_ljs -lr 1e-4 --epochs 1000 -bs 10 --segment-length  8000 --weight-decay 0 --grad-clip-thresh 65504.0 --epochs-per-checkpoint 50 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json --amp --resume-from-last --training-files filelists/tacotron_audio_pitch_train.txt --validation-files filelists/tacotron_audio_pitch_valid.txt
