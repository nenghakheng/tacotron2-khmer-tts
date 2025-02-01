import numpy as np
from scipy.io.wavfile import read
import torch
import os
import librosa


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def load_wav_to_torch(folder_dir ,full_path, sr):

    if folder_dir == 'train':
        full_path = os.path.join('datasets/train_datasets/' ,full_path + '.wav')
    elif folder_dir == 'validation':
        full_path = os.path.join('datasets/validation_datasets/' ,full_path + '.wav')

    sampling_rate, data = read(full_path)

    # Handle resampling
    if sampling_rate != sr:
        data = librosa.resample(data.astype(np.float32), sampling_rate, sr)
        sampling_rate = sr

    audio_tensor = torch.FloatTensor(data.astype(np.float32))

    return audio_tensor, sampling_rate


# def load_filepaths_and_text(filename, split="|"):
#     with open(filename, encoding='utf-8') as f:
#         filepaths_and_text = [line.strip().split(split) for line in f]
#     return filepaths_and_text

# The dataset is in the format of "file_path\t\ttext"
def load_filepaths_and_text(filename, split="\t\t"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
