import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

class AWGNChannel:
    def __init__(self):
        """
        Creates an AWGN channel instance.
        """

    def forward(self, signal, snr_dB):
        """
        Add AWGN noise to the signal (the real part is white noise).
        :param signal: input signal (PyTorch tensor, located on the GPU)
        :param snr_dB: signal-to-noise ratio (in decibels)
        :return: signal with noise (PyTorch tensor on GPU)
        """
        device = signal.device
        snr_linear = 10 ** (snr_dB / 10)
        signal_power = torch.mean(torch.abs(signal) ** 2)    # calculating singal power
        noise_power = torch.sqrt(signal_power / snr_linear)# calculating noise power
        noise_power = noise_power.to(signal.device)
        noise_signal = torch.randn(*signal.shape)# generate noise signal
        noise_signal = noise_signal.to(device)
        noise = noise_power * noise_signal
        noisy_signal = signal + noise# add noise to signal
        return noisy_signal# return noisy signal and put it on GPU
