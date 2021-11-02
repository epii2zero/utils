import torch, torchaudio
import torch.nn as nn
import torch.nn.functional as F

class fft_upsampler(nn.Module):
    """
        torch upsampler expected to work on GPU
        Algorithm: torch.fft > frequency bin zero padding > torch.ifft
    """
    def __init__(self):
        super(fft_upsampler, self).__init__()
        
    def forward(self, waveform: torch.Tensor, scale_factor: int) -> torch.Tensor:
        """
            Input:
                waveform(Tensor): Tensor of real-valued audio (..., time).
            Returns:
                waveform(Temsor): Tensor of resampled real-valued audio (..., time).
        """
        X = torch.fft.rfft(waveform, dim=-1)
        Y = F.pad(X, (0,waveform.shape[-1]//2*(scale_factor-1)), mode='constant', value=0) * scale_factor
        y = torch.fft.irfft(Y, dim=-1)
        return y