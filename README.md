# music_performer_bot
is a AI driven end-to-end musical instrument performer bot, based on a 3-step approach, using state of the art NLP technology:
1) Melody prediction using Transformer Models based on [musicautobot](https://github.com/bearpelican/musicautobot)
2) Mel Spectogram prediction based on [tacotron](https://github.com/NVIDIA/tacotron2)
3) Spectrogram to audio transformation using [waveglow](https://github.com/NVIDIA/waveglow)

Thanks to recent advances in machine learned NLP model, music_performer_bot is able to create realistic sounding instrument samples by just specifying a chord sequence (seq2seq model), or a primer-melody.

Example of a little Song entirely created by samples of music_performer_bot: [beeing hermaniak](https://youtu.be/Uu8OXW23yyc)
(I haven't found a place to store the pretrained saxophone models. let me know if you wanna try them out, I can send them)

## Installation
It is recommended to build a nvidia-docker:
```docker build .```

## Data preparation
raw wav data must stored in following directory structure:
song1
|  wav1_xxxbpm.wav
|  wav2_xxxbpm.wav
|  ....
|  song1.txt

where bpm stands for beats per minute, and song1.txt contains chord information like:
```
Gmaj7 D7 | Gmaj7 B-7 | E-maj7
Am7 ...
```

Run data preparation script:
```
python ./data_preparation.py
```

## Training
See scripts in ```scripts2``` folder

## Inference
```
python ./inference.py --bpm=150 -m "Dm7 G7 C" -o out --waveglow snapshots/waveglow/chechpoint_Waveglow_last.pt --tacotron2 snapshots/tacotron/checkpoint_Tacotron2_last.pt
```
