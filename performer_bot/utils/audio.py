import os, sys, re
import subprocess as sp
from  pathlib import Path
from fastai.basics import get_files
from argparse import ArgumentParser, RawTextHelpFormatter
import os.path
from numpy import array, ma
import aubio
import numpy as np
from scipy import stats


import logging
logger=logging.getLogger('__name__')

def split_audio(filename, out_dir, bars, **kwargs):
    filename = Path(filename)
    out_dir = Path(out_dir)

    if 'bpm' in kwargs:
        bpm = kwargs['pbm']
    else:
        bpm, _ = get_data_from_filename(filename)

    if not bpm:
        raise RuntimeError('please specify beats per minute (bpm)')

    out_dir.mkdir(parents=True, exist_ok=True) 
    out = Path(out_dir) / filename.name 
    split_duration = 4 * 60 * int(bars) /  bpm
    command=f'sox {filename} -r 22050 -b 16 {out} trim 0 {split_duration} : newfile : restart'
    result = sp.run(command.split(),  capture_output=True)


def split_audios(indir, out_dir, bars, **kwargs):
    audios = get_files(indir, extensions='.wav', recurse=True);
    logger.info(f'found {len(audios)} audiofiles, split up in {bars} bar chunks')
    for a in audios:
        logger.debug(f'split audio {a}')
        split_audio(a, out_dir, bars, **kwargs)

def get_data_from_filename(filename):
    bpm = None; bars = None
    bpm_s = re.search('([0-9.]+)bpm', str(filename))
    if bpm_s:
        bpm = float(bpm_s.group(1))

    bars_s = re.search('([0-9]+)bars', str(filename))
    if bars_s:
        bars = int(bpm_s.group(1))

    return bpm, bars

def audio2pitch_seq(audio_file, **kwargs):

    #defaults:
    samplerate = 22050
    hop_s = 256
    win_s = 1024
    tpb = 64

    #import pdb;pdb.set_trace()
    
    bpm, bars = get_data_from_filename(audio_file)
    if 'bpm' in kwargs:
        bpm = kwargs['bpm']
    if 'tpb' in kwargs:
        tpb = kwargs['tpb']   
    
    if not bpm:
        logger.error(f'no bpm for file {audio_file}')
        return np.array([])
    hop_s = int(60 * samplerate / ( bpm * tpb ))

    s = aubio.source(str(audio_file), samplerate, hop_s)
    tolerance = 0.8
    pitch_o = aubio.pitch("yin", win_s, hop_s, s.samplerate)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(tolerance)
    energy = []
    pitches = []
    confidences = []

    # total number of frames read
    total_frames = 0
    while True:
        samples, read = s()
        pitch = pitch_o(samples)[0]
        confidence = pitch_o.get_confidence()
        #f pitch < 100 or pitch > 30:
        pitches += [int(pitch)]
        energy += [np.sqrt(np.mean(samples**2))]
        confidences += [confidence]
        total_frames += read
        if read < hop_s: break

    skip = 1
    pitches = array(pitches[skip:])
    if pitches.size < 0:
        logger.warning(f'Cound not extract any pitches from {audio_file}')
    energy = array(energy[skip:])
    confidences = array(confidences[skip:])
    times = [t * hop_s for t in range(len(pitches))]
    #import pdb;pdb.set_trace()
    cleaned_pitches = pitches
    #cleaned_pitches = ma.masked_where(cleaned_pitches < 0, cleaned_pitches)
     
    seq = cleaned_pitches 
    if np.count_nonzero(seq==0) > (0.8 * seq.size):
        print('  found %s zeros in seq, discard' % np.count_nonzero(seq==0))
        return np.array([])

    k_size = 4
    seq = seq.tolist()
    if len(seq) == 0:
        return np.array([])
    # split in 4-items and calc median 
    out_seq = stats.mode(np.reshape(np.pad(seq, (0, 4-(np.mod(len(seq),4))),mode='constant'), (-1, 4)),1)[0].squeeze()
    
    return out_seq




if __name__ == "__main__":
    parser = ArgumentParser(description='audio utils, split audios', formatter_class=RawTextHelpFormatter)
    parser.add_argument( 'command', help='[sequnce|splie]')
    parser.add_argument( '--inp', '-i', help='audio input dir')
    parser.add_argument( '--out', '-o', help='audio output dir')
    parser.add_argument( '--bars', help='number of bars for split', default=4)
    parser.add_argument('--bpm', type=int, help='beats per minute')
    parser.add_argument('--tpb', type=int, help='ticks per beat')
    parser.add_argument('--type', type=str, help='type of representation')
    parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    if args.command=='split':
        split_audiofiles(args.inp, args.out, args.bars)
    elif args.command == 'sequence':
        seq = audio2pitch_seq(args.inp)
        if seq.size < 2:
            print(f'Warining: file {args.inp} has no sequence')
            sys.exit(1)
        seq_str=[]
        
        # normalize
        for pitch in seq:
            seq_str.append(str(pitch))
        #print (len(seq))
        if seq_str:
            seqs = ','.join(seq_str)
            if args.out:
                with open(args.out, 'w') as fh:
                    print( '[' + seqs + ']', file=fh)
            else:
                print (seqs)
          
