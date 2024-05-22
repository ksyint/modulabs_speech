import torch
import torch.nn as nn

class Lip2Text_Loss(nn.Module):
    def __init__(self, lip2text_model: nn.Module, weight_path: str):
        super().__init__()
        self.lip2text = lip2text_model
        self.lip2text.load_state_dict(torch.load(weight_path))
        for param in self.lip2text.parameters():
                param.requires_grad = False

    def forward(self, inputs, input_lengths, targets):
        loss, loss_ctc, loss_att, acc = self.lip2text(inputs, input_lengths, targets)
        return loss, loss_ctc, loss_att, acc