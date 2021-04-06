#!/bin/sh
#SBATCH --time=48:00:00
#SBATCH --mem=120G
#SBATCH --gres=gpu:2
#SBATCH --job-name=tts_train_De
#SBATCH --output=./egs/egs_bghmk_fix2/train.log
source ~/.bashrc # to make 'conda' available as a command line
source /home/hbauerec/dev/ttt/music_performer_bot/set_env_oda2.sh
python Tacotron2/train.py -m WaveGlow -o snapshots/waveglow/ -lr 1e-4 --epochs 1000 -bs 10 --segment-length  8000 --weight-decay 0 --grad-clip-thresh 65504.0 --epochs-per-checkpoint 50 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json --amp --resume-from-last --training-files filelists/tacotron_audio_pitch_train.txt --validation-files filelists/tacotron_audio_pitch_valid.txt

#python Tacotron2/train.py --n_gpus=2 -o ./snapshots/tacotron -m Tacotron2 --epochs=500 -lr 1e-3 -bs 8 --training-files=filelists/tacotron_mel_pitch_train.txt --validation-files=filelists/tacotron_mel_pitch_valid.txt --load-mel-from-disk --resume-from-last
