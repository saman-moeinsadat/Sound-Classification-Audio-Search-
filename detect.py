import os
import torch
from moviepy.editor import *
import numpy as np
from train import model_prepare
from util import extract_code_label, wave_to_array
from pathlib import Path
import librosa


# The audio_search function takes a file as input and segments
# it and returns a list containg classification for each frame.
# the length of ach segment is 'frame_length' and in seconds.
# The file could be .wav or any video format.

def audio_search_classifier(
    file, sample_rate=44100, block_length=3, n_mels=128, hop_length=512,
    block_hop_length_in_sec=1
):

    # Extracting the name of the file.
    name = file.split('/')[-1].split('.')[0]
    n_frames = int(block_length * sample_rate / hop_length) + 1

    # Block-length in number of samples.
    win_length = sample_rate * block_length
    if file.endswith('wav'):
        audio_series, _ = librosa.load(file, sr=sample_rate)
    else:
        # extracting the audio from video and write it to wave file
        audio_obj = AudioFileClip(file, fps=sample_rate)
        audio_obj.write_audiofile(
            '/'.join(file.split('/')[:-1])+('/%s.wav' % name)
        )
        audio_series, _ = librosa.load(name+'.wav', sr=sample_rate)

    # zero-padding in order to have complete frames.
    n_last = len(audio_series) % win_length
    audio_series = np.pad(
        audio_series, (0, win_length - n_last),
        'constant', constant_values=(0, 0)
    )

    block_hop_length = block_hop_length_in_sec * sample_rate

    number_of_blocks = int(
        len(audio_series) / (block_hop_length)
    ) - 1

    # Building the input array.
    df_tensor = np.zeros(
        shape=(number_of_blocks, 1, n_mels, n_frames)
    )
    for idx in range(number_of_blocks):
        sub_serie = np.array(
            audio_series[
                idx*block_hop_length: (idx*block_hop_length)+win_length
            ]
        )
        df_tensor[idx] = np.array(
            [wave_to_array(sub_serie)]
        )
    df_tensor = torch.from_numpy(df_tensor).float()

    # Load and prepare the model.
    _, cls = extract_code_label()
    model = model_prepare(len(cls))
    model.load_state_dict(
        torch.load('/'.join(file.split('/')[:-1])+'/model_weights.pt')
    )
    model.eval()
    outputs = model(df_tensor)
    _, preds = torch.max(outputs, 1)

    return [cls[i] for i in preds]


# detector_frame function reads audio files from a directory
# and returns a list containg detections coresponding to each
# file in that directory


def detector_frame(
    path, sample_rate=44100, hop_length=512, block_length=3, n_mels=128
):
    file_list = sorted(os.listdir(path))
    n_frames = int(block_length * sample_rate / hop_length) + 1
    in_array = np.zeros(shape=(len(file_list), 1, n_mels, n_frames))
    _, cls = extract_code_label()
    n = 0
    for file in file_list:
        in_array[n] = np.array(
            [wave_to_array('%s/%s' % (path, file))]
        )
        n += 1
    model = model_prepare(len(cls))
    model.load_state_dict(torch.load(
        '/'.join(path.split('/')[:-1])+'/model_weights.pt'
    ))
    model.eval()
    outputs = model(torch.from_numpy(in_array).float())
    _, preds = torch.max(outputs, 1)
    return [cls[i] for i in preds]


if __name__ == "__main__":
    PATH = str((Path(__file__).parent).resolve())
    frame_detections = detector_frame(PATH+'/detect_samples')
    video_classification = audio_search_classifier('~/path_to_video/audiofile')
