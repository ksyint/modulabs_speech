import torch
import torch.nn as nn
import torch.nn.functional as F
# from audio.wav2vec import Wav2Vec
# from generator.PCL_generator import Generator

class SelfLearn(nn.Module):
    def __init__(self, audio_encoder: nn.Module, generator: nn.Module):
        super().__init__()
        self.audio_encoder = audio_encoder
        for param in self.audio_encoder.parameters():
                param.requires_grad = False
        self.fc = nn.Linear(1024, 2048)
        self.layernorm = nn.LayerNorm(2048)
        self.genertor = generator

    def forward(self, input_values, attention_masks):
        logits = self.audio_encoder(input_values, attention_masks)
        feature = self.layernorm(self.fc(logits))
        batch, length, feature_shape = feature.shape
        image = self.genertor(feature.reshape(-1, feature_shape))
        image = F.interpolate(image, size=(88, 88), mode='bilinear', align_corners=False)
        _, c, h, w = image.shape
        image = image.reshape(batch, length, c, h, w)
        return image

if __name__ == "__main__":
    model = SelfLearn(Wav2Vec(), Generator())
    image = model(torch.rand([1, 392400]), torch.ones([1, 392400]))
    print(image.shape)