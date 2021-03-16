from argparse import ArgumentParser, RawTextHelpFormatter

from performer_bot.utils.audio import split_audios
from performer_bot.musicautobot.music_transformer import MusicItem, MusicDataBunch
from pathlib import Path
#from Tacotron2.preprocess_audio2mel import audio2mel
from fastai.data_block import get_files
import random
import logging
logger=logging.getLogger('__name__')
import subprocess as sp

data_filter=['bh']
raw_data_dir=Path('/home/hbauerec/data/de/raw/bh')
out_data_dir=Path('./data/')
data = MusicDataBunch.empty(out_data_dir)
vocab = data.vocab


def prepare_tacotron_data(**kwargs):
    wav_dir = Path('data/split_bars/wav/')
    mel_dir = Path('data/split_bars/mel/')
    wav_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)

    if not any(wav_dir.iterdir()) or not kwargs['lazy']:
        split_audios(raw_data_dir, 'data/split_bars/wav/', 2)
        files = get_files('data/split_bars/wav/',  extensions='.wav', recurse=True);
        with open('filelists/tacotron_audio_pitch_all.txt', 'w') as fh:
            for f in files:
                pitch_arr=MusicItem.from_audio(f, vocab).to_pitch_arr()
                if len(pitch_arr) > 32:
                    fh.write(f"{f}|{pitch_arr}\n")
                else:
                    print(f'skip file {f}, only {len(pitch_arr)} frames')

    if not any(Path('filelists').iterdir()) or not kwargs['lazy']:
        #import pdb;pdb.set_trace()
        file_list = Path(f'filelists/tacotron_audio_pitch_all.txt').read_text().split('\n')
        random.shuffle(file_list)
        spl={'valid': (0,50) , 'test': (50,70), 'train': (70,len(file_list)) }
        for set in ['valid', 'test', 'train']:
            Path(f'filelists/tacotron_audio_pitch_{set}.txt').write_text('\n'.join(file_list[spl[set][0]:spl[set][1]]).strip('\n'))
            Path(f'filelists/tacotron_mel_pitch_{set}.txt').write_text(Path(f'filelists/tacotron_audio_pitch_{set}.txt')  \
                .read_text().replace(r'.wav','.bt').replace(r'/wav/','/mel/'))
        #import pdb;pdb.set_trace()
        Path(f'filelists/tacotron_mel_pitch_all.txt').write_text(Path(f'filelists/tacotron_audio_pitch_all.txt') \
                .read_text().replace(r'.wav','.bt').replace(r'/wav/','/mel/'))
    if not any(Path('data/split_bars/mel').iterdir()) or not kwargs['lazy']:
        cmd = f'python Tacotron2/preprocess_audio2mel.py --wav-files filelists/tacotron_audio_pitch_all.txt --mel-files filelists/tacotron_mel_pitch_all.txt'
        logger.debug(cmd)
        sp.run(cmd.split()) 

if __name__ == "__main__":
    parser = ArgumentParser(description='audio utils, split audios', formatter_class=RawTextHelpFormatter)
    parser.add_argument( '--inp', '-i', help='audio input dir')
    parser.add_argument( '--out', '-o', help='audio output dir')
    parser.add_argument( '--lazy', action="store_true", help='only compute whats missing')
    parser.add_argument( '--bars', help='number of bars for split', default=4)
    parser.add_argument('--bpm', type=int, help='beats per minute')
    parser.add_argument('--tpb', type=int, help='ticks per beat')
    parser.add_argument('--type', type=str, help='type of representation')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger.info("prepare tacotron data -> split audio into 2 bars chunks and transcribe audio")
    prepare_tacotron_data(**vars(args))
