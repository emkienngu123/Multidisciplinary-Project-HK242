import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperProcessor


class VoiceCommandWhisper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_out = cfg['model']['d_out']

        # Load pre-trained Whisper Tiny model
        self.feature_extractor = WhisperModel.from_pretrained("openai/whisper-tiny")

        # Load WhisperProcessor for preprocessing
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

        # Classification head for fine-tuning
        self.cls_head = nn.Linear(384, self.d_out)

    def forward(self, x, sampling_rate=16000):
        # Preprocess input using WhisperProcessor
        x = x.cpu().numpy()  # Convert to numpy for processor compatibility
        inputs = self.processor(x, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(x.device)  # Shape: (batch_size, num_frames, feature_dim)

        # Extract features using Whisper
        with torch.no_grad():
            outputs = self.feature_extractor.encoder(input_features)  # Shape: (batch_size, seq_len, 384)
            features = outputs.last_hidden_state

        # Pooling (mean over the sequence length)
        features = torch.mean(features, dim=1)  # Shape: (batch_size, 384)

        # Pass features through the classification head
        x = self.cls_head(features)
        return x


def build_model(cfg):
    return VoiceCommandWhisper(cfg)