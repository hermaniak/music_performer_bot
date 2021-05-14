#!/bin/sh

python Tacotron2/train.py -m WaveGlow -o snapshots/waveglow_ljs -lr 1e-4 --epochs 1000 -bs 10 --segment-length  8000 --weight-decay 0 --grad-clip-thresh 65504.0 --epochs-per-checkpoint 50 --cudnn-enabled --cudnn-benchmark --log-file nvlog.json --amp --resume-from-last --training-files filelists/tacotron_audio_pitch_train.txt --validation-files filelists/tacotron_audio_pitch_valid.txt
