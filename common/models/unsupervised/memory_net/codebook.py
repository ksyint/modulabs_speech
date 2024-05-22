
import torch
import torch.nn as nn
import torch.nn.functional as F

class Codebook(nn.Module):
    def __init__(self, slot=112, embedding=512, requires_grad=True):
        super(Codebook, self).__init__()
        self.slot = slot
        self.v_embd = nn.Linear(dim, 512)
        self.weight_matrix = nn.Parameter(torch.rand((slot, embedding), requires_grad=requires_grad))
    def forward(self, x):
        # 행렬 곱 수행
        weighted_output = torch.matmul(x, self.weight_matrix)

        # Softmax 취하기
        softmax_weights = F.softmax(weighted_output, dim=1)

        # Softmax를 가중치로 사용하여 weighted_matrix를 sum
        aud = softmax_weights @ self.weight_matrix.permute(1, 0)
        if not self.inference:   
            contrastive_loss = 0.5 * torch.abs(torch.eye(self.slot).to(value.device) - torch.matmul(value_norm, value_norm.transpose(0, 1))).sum()    #n_slot,n_slot
            recon_loss = torch.abs(1.0 - F.cosine_similarity(aud, value.detach(), 1))  #BS
            recon_loss = recon_loss.mean(0)   #B
            return (aud, recon_loss, contrastive_loss)
        return aud

if __name__ == "__main__":
    module = Codebook()
    print(module(torch.rand(3, 2048)))