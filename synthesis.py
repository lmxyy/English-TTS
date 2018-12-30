#!/usr/bin/env python3
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm
from wavenet_vocoder.hparams import hparams
from wavenet_vocoder.synthesis import wavegen
from wavenet_vocoder.train import build_model


def run_proc(idx, text, mel):
    print("\n", idx, text)
    mel_path = os.path.join("./", mel)
    c = np.load(mel_path)
    if c.shape[1] != hparams.num_mels:
        np.swapaxes(c, 0, 1)
    # Range [0, 4] was used for training Tacotron2 but WaveNet vocoder assumes [0, 1]
    c = np.interp(c, (0, 4), (0, 1))

    # Generate
    waveform = wavegen(model, c=c, fast=True, tqdm=tqdm)

    waveforms.append(waveform)
    wavfile.write('output/%d.wav' % idx, hparams.sample_rate, waveform)


if __name__ == '__main__':
    command = "PYTHONPATH=$PYTHONPATH:./Tacotron-2 /Users/limuyang/anaconda3/envs/tacotron2/bin/python ./Tacotron-2/synthesize.py --model='Tacotron' --mode='eval' --hparams='symmetric_mels=False,max_abs_value=4.0,power=1.1,outputs_per_step=1' --text_list=./text_list.txt"
    print(command, file=sys.stderr)
    # os.system(command)
    print('Mel-spectrogram prediction by Tacoron2 finish.', file=sys.stderr)

    wn_preset = 'wavenet_vocoder/pretrained/20180510_mixture_lj_checkpoint_step000320000_ema.json'
    wn_checkpoint_path = 'wavenet_vocoder/pretrained/20180510_mixture_lj_checkpoint_step000320000_ema.pth'
    with open(wn_preset) as f:
        hparams.parse_json(f.read())
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = build_model().to(device)
    print("Load checkpoint from {}".format(wn_checkpoint_path))
    if use_cuda:
        checkpoint = torch.load(wn_checkpoint_path)
    else:
        checkpoint = torch.load(wn_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])

    with open("tacotron_output/eval/map.txt") as f:
        maps = f.readlines()
    maps = list(map(lambda x: x[:-1].split("|"), maps))
    # filter out invalid ones
    maps = list(filter(lambda x: len(x) == 2, maps))

    print("List of texts to be synthesized")
    for idx, (text, _) in enumerate(maps):
        print(idx, text)

    waveforms = []
    os.makedirs('output', exist_ok=True)

    pool = Pool(2)
    for idx, (text, mel) in enumerate(maps):
        pool.apply_async(run_proc, args=(idx, text, mel))
    pool.close()
    pool.join()
