import numpy as np
import os
import librosa

def file_length_statistic(path, object_type=['.wav']):
    file_list = os.listdir(path)
    max_len = 0
    max_name = None
    min_len = -1
    min_name = None
    total_len_sec = 0
    
    file_iter_bar = tqdm(file_list)
    for file in file_iter_bar:
        file_name = os.path.splitext(file)[0]
        file_type = os.path.splitext(file)[1]
        if file_type not in object_type:
            print('wrong type:', file_name)
            continue
        wav, sr = librosa.load(os.path.join(path, file), sr=None)
        wav_len = len(wav)
        if wav_len > max_len:
            max_len = wav_len
            max_name = file
        if (wav_len < min_len) or (min_len == -1):
            min_len = wav_len
            min_name = file
        total_len_sec += wav_len / sr
    return max_len, max_name, min_len, min_name, total_len_sec