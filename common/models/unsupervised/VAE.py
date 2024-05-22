import torch
import torch.nn.functional as F
from torch import nn

class CodeBVAE(nn.Module):
    def __init__(self, encoder:torch.nn.Module, codebook:torch.nn.Module , decoder:torch.nn.Module):
        super(CodeBVAE, self).__init__()
        self.encoder = encoder 

        self.fc = nn.Sequential(nn.Linear(2048, 2048),
                                         nn.ReLU(),
                                         nn.Linear(2048,512),
                                         nn.BatchNorm1d(512))
        self.codebook = codebook
        self.decoder = decoder #Generator(d_model=2048)

    def forward(self, input_):
        feature = self.encoder(input_)
        fc_feature = self.fc(feature)
        vq_output = self.codebook(fc_feature)
        if isinstance(vq_output, tuple):
            vq_feature, recon_loss, contrastive_loss = vq_output
        else:
            vq_feature = vq_output
            recon_loss, contrastive_loss = None, None
        output = self.decoder(vq_feature)
        return output, recon_loss, contrastive_loss

if __name__=="__main__":
    from networks import FaceCycleBackbone, Generator
    from memory_net.memory import Memory
    from torchsummary import summary
    encoder = FaceCycleBackbone()
    decoder = Generator(d_model=512)
    memory = Memory()
    model = CodeBVAE(encoder, memory, decoder)
    output = model(torch.rand((2,3,64, 64)))
    print(output[0].shape)
    print(encoder(torch.rand((2,3,64, 64))).shape)