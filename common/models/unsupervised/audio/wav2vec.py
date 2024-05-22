import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC

class Wav2Vec(nn.Module):
    def __init__(self, path:str = "masoudmzb/wav2vec2-xlsr-multilingual-53-fa"):
        super().__init__()
        self.model = Wav2Vec2ForCTC.from_pretrained(path)
        self.model.dropout = nn.Identity()
        self.model.lm_head = nn.Identity()
        
    def forward(self, input_values, attention_masks):
        logits = self.model(input_values, attention_mask=attention_masks).logits
        return logits

if __name__ == "__main__":
    model = Wav2Vec()
    logits = model(torch.rand([1, 392400]), torch.ones([1, 392400]))
    print(logits.shape)