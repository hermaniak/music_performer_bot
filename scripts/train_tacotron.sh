#!/bin/sh
python Tacotron2/train.py -o ./snapshots/tacotron -m Tacotron2 --epochs=1000 -lr 1e-3 -bs 8 --training-files=filelists/tacotron_mel_pitch_train.txt --validation-files=filelists/tacotron_mel_pitch_valid.txt --load-mel-from-disk --resume-from-last

