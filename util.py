import librosa
import librosa.display
import librosa.effects
import librosa.feature
import numpy as np
from pathlib import Path
import os
import random
from numpy.random import randint
import soundfile as sf


# The wave to array function takes as inout either path to the
# wave file or audio time series in numpy format.
# It pre-processes the data(Normalization, pre-emphasis) and extracts
# 128 Mel features for each 700 Ms SFFT frames for 44.1Kz sample rate,
# hence the dimnetion of out put (128, 259).
# In the end it is converted to db, to produce log-scaled Mel spectrogram.


def wave_to_array(data, sample_rate=44100, n_mels=128, sample_length=3):
    # Checking the input and read the data.

    if isinstance(data, str):
        audio_series, sample_rate = librosa.load(data, sr=sample_rate)
    elif isinstance(data, np.ndarray):
        audio_series = data
    else:
        return """
                The input type must be either path to the
                sound file of numpy ndarray.
                """

    number_of_samples = len(audio_series)

    # Zero pads the data to have windows-length equal to
    # sample_length.

    fix_number_of_samples = int(sample_rate*sample_length)
    if number_of_samples <= fix_number_of_samples:
        audio_series = np.pad(audio_series, (
            0, fix_number_of_samples - number_of_samples),
            'constant', constant_values=(0, 0)
        )
    else:
        audio_series = audio_series[:fix_number_of_samples-1]

    # Normalization:

    audio_series_norm = librosa.util.normalize(audio_series)

    # Pre-emphasis:

    audio_series_emp = librosa.effects.preemphasis(audio_series_norm)

    # Mel Spectrogram: extracts 128 Mel features for 259 frames ==>
    # numpy ndarray with (128, 259) shape:

    mel = librosa.feature.melspectrogram(
        audio_series_emp,
        sr=sample_rate,
        n_mels=n_mels
    )

    return librosa.power_to_db(mel, ref=np.max)


# dataset_clean_number function, clean the the dataset
# from backup-saved files from DAWs like Reaper and
# returns the dataset length.


def dataset_clean_number(path):
    dir_list = sorted(os.listdir(path))
    dataset_length = 0
    for dir in dir_list:
        sub_dir = os.listdir(path+'/'+dir)
        for file in sub_dir:
            if file.endswith(".wav"):
                dataset_length += 1
            else:
                # Removes the unwanted files.
                os.remove(path+'/'+dir+'/'+file)
    return dataset_length


# Reads the waves from 'data' directory that is divided in 12 classes,
# exracts the Mel features and the label for each sample and stores
# and saves them into numpy ndarray, one for labels and one for data.

def dir_to_df(
    dst_length, n_mels=128,
    path=str((Path(__file__).parent / "data").resolve()),
    sample_rate=44100, hop_length=512, frame_length=3
):

    dir_list = sorted(os.listdir(path))
    label = []

    # numpy empy array with shape:
    # (dataset-length, 1, number of mel features, number of frames)

    n_frames = int(frame_length * sample_rate / hop_length) + 1

    df_tensor = np.zeros(shape=(dst_length, 1, n_mels, n_frames))

    n = 0
    for item in dir_list:
        sub_dir_list = os.listdir('%s/%s' % (path, item))
        for sub_item in sub_dir_list:
            if sub_item.endswith(".wav"):
                label.append(item)
                df_tensor[n] = np.array(
                    [wave_to_array('%s/%s/%s' % (path, item, sub_item))]
                )
                n += 1

    label = np.array(label)
    np.save('data.npy', df_tensor)
    np.save('label.npy', label)


# The extract_code_label function encodes the labels and returns
# the encoded labels with coresponding classes.

def extract_code_label(
    label_path=str((Path(__file__).parent / 'label.npy').resolve())
):
    label = np.ndarray.tolist(np.load(label_path))
    cls = sorted(list(set(label)))
    for i in range(len(cls)):
        for j in range(len(label)):
            if cls[i] == label[j]:
                label[j] = i
    return np.array(label, dtype=int), np.array(cls)


# data_augmentation takes in the wave samples of our dataset
# and solve the problem of im-balanced or small dataset
# by producing new sample waves through stretching or pitch-shifting
# of the original samples. It balances the dataset for 'class_number'
# numbers of samples per class.


def data_augmentation(path, class_number=300, sample_rate=44100):
    dir_list = sorted(os.listdir(path))

    # Stretch factors:
    stretches = [
        0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.1, 1.15, 1.25, 1.35, 1.4, 1.5
    ]

    # Pitch-shift factors:
    n_steps = [-12, -10, -9, -8, -7, -6, -4, -2, 2, 4, 6, 7, 8, 9, 10, 12]

    for directory in dir_list:
        file_list = os.listdir('%s/%s' % (path, directory))
        idx_list = []
        n = 0

        # Saves the augmented data in new directory for further
        # examination.
        if len(file_list) < class_number:

            # A list containing indices of original files in the directory
            # and length of the number of augmented samples needed for a
            # specific class.

            res = randint(0, len(file_list), class_number - len(file_list))
            for file_idx in res:
                n += 1
                audio_series, sample_rate = librosa.load(
                    '%s/%s/%s'   (path, directory, file_list[file_idx]),
                    sr=sample_rate
                )

                # Random selection of algorithms for data augmentation.
                # Another way for augmenting sound files is convolving them
                # with different background noise and/or with different
                # reverberation(Hall) impulse responses.
                # For this project the task is to classify the dry samples of
                # instruments in a drum-kit and does not contain instances of
                # background noise or reverberation.

                if file_idx % 2 == 0:
                    if file_idx not in idx_list:
                        audio_augmented = librosa.effects.pitch_shift(
                            audio_series, sample_rate,
                            n_steps=random.choice(n_steps),
                            bins_per_octave=24
                        )
                        idx_list.append(file_idx)
                    else:
                        audio_augmented = librosa.effects.time_stretch(
                            audio_series,
                            random.choice(stretches)
                        )
                elif file_idx % 2 != 0:
                    if file_idx not in idx_list:
                        audio_augmented = librosa.effects.time_stretch(
                            audio_series,
                            random.choice(stretches)
                        )
                        idx_list.append(file_idx)
                    else:
                        audio_augmented = librosa.effects.pitch_shift(
                            audio_series, sample_rate,
                            n_steps=random.choice(n_steps),
                            bins_per_octave=24
                        )

                sf.write(
                        '%s/%s/%d.wav' % (path, directory, n),
                        audio_augmented, sample_rate, subtype='PCM_24'
                        )


# Building the un-labelled dataset for the purpose of
# semi-supervised learning.


def dir_unlabelled_to_fd(
    n_mels=128,
    path=str((Path(__file__).parent / "data_unlabelled").resolve()),
    sample_rate=44100, hop_length=512, frame_length=3
):

    # Clean data from none wave files:
    for file in os.listdir(path):
        if not file.endswith(".wav"):
            os.remove(path+'/'+file)

    len_dst = len(os.listdir(path))
    n_frames = int(frame_length * sample_rate / hop_length) + 1
    data_unlabelled = np.zeros(shape=(len_dst, 1, n_mels, n_frames))
    n = 0
    for file in os.listdir(path):
        data_unlabelled[n] = np.array(
            [wave_to_array('%s/%s' % (path, file))]
        )
        n += 1

    np.save('data_unlabelled.npy', data_unlabelled)


if __name__ == "__main__":
    PATH = str((Path(__file__).parent).resolve())
    data_augmentation('%s/data/' % PATH)
    dst_length = dataset_clean_number('%s/data/' % PATH)
    dir_to_df(dst_length)
    dir_unlabelled_to_fd()
