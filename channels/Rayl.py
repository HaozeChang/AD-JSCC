import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

class RayleighChannel:
    def __init__(self):
        """
        Create a Rayleigh fading channel instance.
        """
        pass

    def forward(self, signal, snr_dB):
        """
        Propagate the signal through a Rayleigh fading channel and add the corresponding noise.
        :param signal: input signal (PyTorch tensor, located on the GPU)
        :param snr_dB: signal-to-noise ratio (in decibels)
        :return: signal with noise after Rayleigh fading (PyTorch tensor on GPU)
        """
        device = signal.device
        snr_linear = 10 ** (snr_dB / 10)
        
        # Rayleigh fading factor
        real_part = torch.randn(signal.shape, device=device)
        imag_part = torch.randn(signal.shape, device=device)
        rayleigh_fading = torch.sqrt(real_part ** 2 + imag_part ** 2) / torch.sqrt(torch.tensor(2.0, device=device))
        
        # add factor to signal
        faded_signal = signal * rayleigh_fading
        
        # signal power and noise power
        signal_power = torch.mean(faded_signal.abs() ** 2)
        noise_power = torch.sqrt(signal_power / snr_linear)
        noise = noise_power * torch.randn(signal.shape, device=device)
        
        # add faded noise to signal 
        noisy_signal = faded_signal + noise
        return noisy_signal