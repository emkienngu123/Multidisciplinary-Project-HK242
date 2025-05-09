import os
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torchaudio.transforms as T
from torchvision.transforms import Compose


class RandomVol:
    def __init__(self, min_gain=0.3, max_gain=1.0, rate=0.5):
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.rate = rate

    def __call__(self, signal):
        if random.random() < self.rate:
            gain = random.uniform(self.min_gain, self.max_gain)
            vol_transform = T.Vol(gain=gain).to(signal.device)  # Ensure the transform is on the correct device
            return vol_transform(signal)
        return signal


class RandomFrequencyMasking:
    def __init__(self, max_freq_mask_param=30, rate=0.5):
        self.max_freq_mask_param = max_freq_mask_param
        self.rate = rate

    def __call__(self, signal):
        if random.random() < self.rate:
            freq_mask_param = random.randint(0, self.max_freq_mask_param)
            freq_mask_transform = T.FrequencyMasking(freq_mask_param=freq_mask_param).to(signal.device)
            return freq_mask_transform(signal)
        return signal


class RandomTimeMasking:
    def __init__(self, max_time_mask_param=50, rate=0.5):
        self.max_time_mask_param = max_time_mask_param
        self.rate = rate

    def __call__(self, signal):
        if random.random() < self.rate:
            time_mask_param = random.randint(0, self.max_time_mask_param)
            time_mask_transform = T.TimeMasking(time_mask_param=time_mask_param).to(signal.device)
            return time_mask_transform(signal)
        return signal

class RandomTimeShift:
    def __init__(self, max_shift: int, rate=0.5):
        self.max_shift = max_shift
        self.rate = rate

    def __call__(self, signal):
        if random.random() < self.rate:
            shift = random.randint(-self.max_shift, self.max_shift)
            if shift > 0:
                signal = torch.cat((signal[:, shift:], signal[:, :shift]), dim=1)
            elif shift < 0:
                signal = torch.cat((signal[:, shift:], signal[:, :shift]), dim=1)
        return signal


class VoiceCommandDataset(Dataset):
    def __init__(self,
                 annotations_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device,
                 augmentations=None,
                 train=True):
        self.annotations = pd.read_csv(os.path.join(audio_dir,annotations_file))
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.augmentations = augmentations
        self.train = train

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)

        # Apply augmentations if provided
        if self.augmentations:
            signal = self.augmentations(signal)
        #signal = self.transformation(signal).squeeze(0).transpose(0,1).detach()
        signal = signal.squeeze(0)

        if self.train:
            label = self._get_audio_sample_label(index)
            return signal, label
        else:
            return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        data_dir = f"{self.annotations.iloc[index, 0]}.mp3"
        path = os.path.join(self.audio_dir, data_dir)
        return path

    def _get_audio_sample_label(self, index):
        if self.train:
            return self.annotations.iloc[index, 1]
        else:
            raise ValueError("Labels are not available in inference mode (train=False).")
def build_audio_transformation(cfg):
    mel_spectrogram = T.MFCC(
        sample_rate=cfg['dataset']['sample_rate'],
        n_mfcc=cfg['dataset']['transformation']['n_mfcc'],
        melkwargs={
            "n_fft": cfg['dataset']['transformation']['n_fft'],
            "hop_length": cfg['dataset']['transformation']['hop_length'],
            "n_mels": cfg['dataset']['transformation']['n_mels']
        }
    )
    return mel_spectrogram
def build_augmentation(cfg):
    augmentations = []
    if cfg['dataset']['augmentation']['random_vol']:
        augmentations.append(RandomVol(cfg['dataset']['augmentation']['random_vol']['min_gain'],
                                       cfg['dataset']['augmentation']['random_vol']['max_gain'],
                                       cfg['dataset']['augmentation']['rate']))
    if cfg['dataset']['augmentation']['random_freq_mask']:
        augmentations.append(RandomFrequencyMasking(cfg['dataset']['augmentation']['random_freq_mask']
                                                    ['max_freq_mask_param'],
                                                    cfg['dataset']['augmentation']['rate']))
    if cfg['dataset']['augmentation']['random_time_mask']:
        augmentations.append(RandomTimeMasking(cfg['dataset']['augmentation']['random_time_mask']
                                               ['max_time_mask_param'],
                                               cfg['dataset']['augmentation']['rate']))
    if cfg['dataset']['augmentation']['random_time_shift']:
        augmentations.append(RandomTimeShift(cfg['dataset']['augmentation']['random_time_shift']['max_shift'],
                                             cfg['dataset']['augmentation']['rate']))
    return Compose(augmentations)


def build_dataset(cfg, anno_file, device, training=False):
    transformation = build_audio_transformation(cfg)
    augmentations = None
    if training and anno_file!='val.csv':
        augmentations = build_augmentation(cfg)
    dataset = VoiceCommandDataset(
        annotations_file=anno_file,
        audio_dir=cfg['dataset']['audio_dir'],
        transformation=transformation,
        target_sample_rate=cfg['dataset']['sample_rate'],
        num_samples=cfg['dataset']['num_samples'],
        device=device,
        augmentations=augmentations,
        train=training
    )
    return dataset