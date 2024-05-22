import torch
from torch import nn
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, radius=16.0, n_slot=112, n_head=8, dim=512, diff_aud_vid=False, inference=False):
        super().__init__()
        self.inference = inference
        self.dav = diff_aud_vid

        self.head = n_head
        self.slot = n_slot

        self.value = nn.Parameter(torch.Tensor(n_slot, 512), requires_grad=True)
        nn.init.normal_(self.value, 0, 0.5)

        self.v_embd = nn.Linear(dim, 512)

        self.radius = radius
        self.softmax = nn.Softmax(1)

    def forward(self, value):
        embd_value = self.v_embd(value)
        value_norm = F.normalize(self.value, dim=1) #n_slot,512
        value_sim = F.linear(F.normalize(embd_value, dim=1), value_norm) #BS, n_slot
        value_add = self.softmax(self.radius * value_sim)

        aud = torch.matmul(value_add, self.value)   #BS,512

        if not self.inference:   
            contrastive_loss = 0.5 * torch.abs(torch.eye(self.slot).to(value.device) - torch.matmul(value_norm, value_norm.transpose(0, 1))).sum()    #n_slot,n_slot
            recon_loss = torch.abs(1.0 - F.cosine_similarity(aud, value.detach(), 1))  #BS
            recon_loss = recon_loss.mean(0)   #B
            return (aud, recon_loss, contrastive_loss)
        return aud

if __name__ == "__main__":
    model = Memory().cuda()

    input_ = torch.rand((5, 512)).cuda()
    aud, recon_loss, contrastive_loss = model(input_)
    print(aud.shape, recon_loss, recon_loss.shape, contrastive_loss.shape)
    