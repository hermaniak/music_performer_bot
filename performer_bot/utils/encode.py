
import os

from music21 import *
from aubio import source, notes
import numpy as np
import logging
from argparse import ArgumentParser, RawTextHelpFormatter, FileType

logger=logging.getLogger(__name__)

BPB = 4 # beats per bar
TIMESIG = f'{BPB}/4' # default time signature
PIANO_RANGE = (21, 108)
VALTSEP = -1 # separator value for numpy encoding
VALTCONT = -2 # numpy value for TCONT - needed for compressing chord array

def encode_melody_from_audio(wav_file):
    samplerate = 22050
    win_s = 256  # fft size
    hop_s = 128  # hop size

    s = source(wav_file, samplerate, hop_s)
    samplerate = s.samplerate

    notes_o = notes("default", win_s, hop_s, samplerate)

    ticks_per_beat = 4
    bpm = 110 # default midi tempo

    def frames2tick(frames, samplerate=samplerate):
        return  int((frames * ticks_per_beat * bpm) / (samplerate * 60))
    
    # total number of frames read
    total_frames = 0
    note={'pitch': VALTSEP, 'tick': 0}
    notes_np=[]

    def correct_duration(pitch, duration):
        if pitch == VALTSEP:
            duration += 1
        return [pitch, duration]

    def no_separator_at_end(notes_np, note):
        return (notes_np and notes_np[-1][0] != VALTSEP and note['pitch'] != VALTSEP)

    while True:
        samples, read = s()
        new_note = notes_o(samples)
        now = frames2tick(total_frames) 
        
        if new_note[0] != 0:
            #finish last note
            if now - note['tick'] > 0:
                if no_separator_at_end(notes_np, note):
                    notes_np.append([VALTSEP, 1])

                notes_np.append(correct_duration(note['pitch'], now - note['tick']))
            note={'pitch': new_note[0], 'tick': now}
        
        # note end
        if new_note[0] == 0 and new_note[2] != 0:
            if note['pitch'] != VALTSEP and new_note[2] == note['pitch'] and now - note['tick'] > 0:
                if no_separator_at_end(notes_np, note):
                    notes_np.append([VALTSEP, 1])
                notes_np.append(correct_duration(note['pitch'], now - note['tick']))
                # add pause
                note={'pitch': VALTSEP, 'tick': now}
                
        total_frames += read
        if read < hop_s: break
        ## check 
        #for n in notes_np    
    return np.array(notes_np, dtype=int).reshape(-1,2)


def encode_chords_from_txt_file(chords_file):
    chords_txt=""
    if not os.path.exists(chords_file):
        logger.ERROR(f'chord file {chords_file} does not exits')
        return None
    #import pdb;pdb.set_trace() 
    for line in open(chords_file):
        chords_txt = chords_txt + ' | ' + line.strip('\n')
    return encode_chords_from_txt(chords_txt)


def encode_chords_from_txt(chords_txt):
    # expand to 4 chords per bar
    chords=[]
    chords_np=[]
    VALTSEP=-1
    chord_duration = 4 # 4 16th note
    for chord in (chords_txt).split('|'):
        ch_arr = (chord).split()
        for ch in ch_arr:
            try:
                harm = harmony.ChordSymbol(ch.strip())
            except:
                print(f'chord {ch} not valid')
                continue
            for i in range(int(4 / len(ch_arr))):
                chords.append(harm)
                for pitch in harm.pitches:
                    chords_np.append([pitch.midi, chord_duration])
                chords_np.append([VALTSEP, (len(ch_arr) - 1) * chord_duration])

    return np.array(chords_np, dtype=int)


def encode_melody_chords(wav_file, chords_file, output_dir):
    
    m = encode_melody_from_audio(wav_file)
    c = encode_chords_from_txt_file(chords_file)
    m_c = np.array([m, c], dtype=object)
    #import pdb;pdb.set_trace()
    fname=os.path.basename(wav_file).replace('.wav','.npy')
    np.save(f'{output_dir}/lm/{fname}', np.array(m)) 
    np.save(f'{output_dir}/s2s/{fname}', m_c)
    print(f'save to {output_dir}/s2s/{fname}') 

def encode_s2s(s2s_numpy_path, save_file):
    #import pdb;pdb.set_trace()
    s2s_numpy_path='data/numpy/s2s/oleo/'
    lm_data_save_name = 'oleo_musicitem_data_save.pkl'
    s2s_data_save_name = 'oleo_multiitem_data_save.pkl'

    save_file = Path(save_file)
    get_files(s2s_numpy_path, extensions='.npy', recurse=True); 
    len(s2s_numpy_files)
    if len(s2s_numpy_files) > 0:
        save_file.parent.mkdir(exist_ok=True, parents=True)
        vocab = MusicVocab.create()
        processors = [S2SFileProcessor(), S2SPartsProcessor()]
        data = MusicDataBunch.from_files(files, path, processors=processors, 
                                          preloader_cls=S2SPreloader, list_cls=S2SItemList, dl_tfms=melody_chord_tfm)
        data.save(s2s_data_save_name)


if __name__ == "__main__":
    ## unit tests and more...
    parser = ArgumentParser(description='encode data into numpy arrays', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--wav_file', '-i', help='inout txt file.')
    parser.add_argument('--chord_file', '-c', help='output txt file.')
    parser.add_argument('--out_dir', '-o', help='output numpy array')
    parser.add_argument('--lazy', '-l', help='use cached data rather recompute all')

    args = parser.parse_args()
     
    encode_melody_chords(args.wav_file, args.chord_file, args.out_dir)
  
