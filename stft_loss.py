import torch

def stft(x, win_len, hop_len, fft_size=None, return_magnitude=True):
    # x: audio waveform (batch_size x audio_length)

    # FFT size initialization
    if fft_size is None:
        fft_size = win_len
    
    # Windowing function
    window = torch.hann_window(win_len, device=x.device).unsqueeze(0)

    # Signal length adjustment
    audio_len = x.size(1)

    if audio_len < win_len:
        x = torch.nn.functional.pad(x, (0, win_len - audio_len))

    if (audio_len - win_len) % hop_len != 0:
        x = torch.nn.functional.pad(x, (0, hop_len - ((audio_len - win_len) % hop_len)))

    # Framing
    audio_len = x.size(1)
    num_frame = (audio_len - win_len) // hop_len + 1

    stft_in = torch.zeros(x.size(0), num_frame, win_len, device=x.device)
    for i in range(num_frame):
        stft_in[:, i] = x[:, hop_len * i:hop_len * i + win_len] * window

    # FFT
    stft_out = torch.rfft(stft_in, signal_ndim=1)

    if return_magnitude:
        stft_out = torch.sqrt(torch.sum(stft_out ** 2, dim=3))

    return stft_out


# Parallel WaveGAN
def stft_loss_combined(ref, syn, win_len, hop_len, fft_size, ratio=1):
    eps = 1e-5
    def stft_loss_sc(stft_ref, stft_syn):
        fro_ref = torch.sqrt(torch.sum(stft_ref ** 2) + eps) + eps
        fro_diff = torch.sqrt(torch.sum((stft_ref - stft_syn) ** 2) + eps) + eps

        loss_sc = fro_diff / fro_ref

        return loss_sc


    def stft_loss_mag(stft_ref, stft_syn):
        eps = 1e-5
        
        log_diff = torch.log(torch.clamp(stft_ref, min=eps)) - torch.log(torch.clamp(stft_syn, eps))
        loss_mag = torch.mean(torch.abs(log_diff))
        
        return loss_mag

    stft_ref = stft(ref, win_len, hop_len, fft_size, return_magnitude=True)
    stft_syn = stft(syn, win_len, hop_len, fft_size, return_magnitude=True)

    loss_sc = stft_loss_sc(stft_ref, stft_syn)
    loss_mag = stft_loss_mag(stft_ref, stft_syn)
    
    print('loss_sc:',loss_sc)
    print('loss_mag:',loss_mag)
    
    loss_total = loss_sc + ratio * loss_mag

    return loss_total


# Neural source filter
def stft_loss_nsf(ref, syn, win_len, hop_len, fft_size, ratio=1):
    def stft_amp_distance(stft_ref, stft_syn):
        eps = 1e-5

        num = torch.log(torch.clamp(torch.sum(stft_ref ** 2, dim=3), eps))
        den = torch.log(torch.clamp(torch.sum(stft_syn ** 2, dim=3), eps))

        distance = torch.mean((num - den) ** 2)

        return distance


    def stft_phase_distance(stft_ref, stft_syn):
        num = torch.sum(stft_ref * stft_syn, dim=3)

        ref_mag = torch.sqrt(torch.sum(stft_ref ** 2, dim=3))
        syn_mag = torch.sqrt(torch.sum(stft_syn ** 2, dim=3))

        distance = torch.mean(1 - num / torch.clamp(ref_mag * syn_mag, 1e-5))

        return distance

    stft_ref = stft(ref, win_len, hop_len, fft_size, return_magnitude=False)
    stft_syn = stft(syn, win_len, hop_len, fft_size, return_magnitude=False)

    loss_amp_distance = stft_amp_distance(stft_ref, stft_syn)
    loss_phase_distance = stft_phase_distance(stft_ref, stft_syn)

    loss_total = loss_amp_distance + ratio * loss_phase_distance

    return loss_total

