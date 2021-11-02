import os
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

def plot_librosa_spectrogram(data, sr, n_fft, hop_length, win_length, window,
                             title, save_dir, figsize, dpi=300,
                             cmap='jet', clim=None):
    """
    
    """
    plt.figure(figsize=figsize)
    stft = librosa.stft(y=data, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    D = librosa.amplitude_to_db(np.abs(stft))
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap=cmap)
    plt.colorbar(format='%+2.0f dB')
    plt.clim(clim)
    plt.title(title)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    if save_dir == None:
        return
    plt.savefig(f'{save_dir}/{title}.png', dpi=dpi)
    
    
def single_librosa_spectrogram(data, sr, n_fft, hop_length, win_length, window,
                               title, dpi=300, cmap='jet'):
    stft = librosa.stft(y=data, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    D = librosa.amplitude_to_db(np.abs(stft))
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap=cmap)
    plt.title(title)
    
    
def plot_plt_histogram(data, bins,
                       title, save_dir, figsize, dpi=300,
                       ):
    plt.figure(figsize=figsize)
    plt.hist(data, bins=bins)
    plt.title(title)
    if save_dir == None:
        return
    plt.savefig(f'{save_dir}/{title}.png', dpi=dpi)
    


