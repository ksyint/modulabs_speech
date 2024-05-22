import torch
import torch.nn as nn

class Generator(torch.nn.Module):
    def __init__(self, d_model:int = 2048, model_path:str =  None):
        super(Generator, self).__init__()
        self.d_model = d_model
        
        self.fc = nn.Linear(d_model,512)
        self.d_model = 512

        up = nn.Upsample(scale_factor=2, mode='bilinear')

        dconv1 = nn.Conv2d(self.d_model, self.d_model//2, 3, 1, 1) # 2*2 512
        dconv2 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 4*4 256
        dconv3 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 16*16 256
        dconv4 = nn.Conv2d(self.d_model//2, self.d_model//2, 3, 1, 1) # 32 * 32 * 256
        dconv5 = nn.Conv2d(self.d_model//2, self.d_model//4, 3, 1, 1) #  64 * 64 *128
        #dconv6 = nn.Conv2d(self.d_model//4, self.d_model//8, 3, 1, 1) # 128 * 128 *32
        dconv7 = nn.Conv2d(self.d_model//4, 3, 3, 1, 1)

        # batch_norm2_1 = nn.BatchNorm2d(self.d_model//8)
        batch_norm4_1 = nn.BatchNorm2d(self.d_model//4)
        batch_norm8_4 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_5 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_6 = nn.BatchNorm2d(self.d_model//2)
        batch_norm8_7 = nn.BatchNorm2d(self.d_model//2)

        relu = nn.ReLU()
        tanh = nn.Tanh()

        self.model = torch.nn.Sequential(relu, up, dconv1, batch_norm8_4, \
                             relu, up, dconv2, batch_norm8_5, relu,
                             up, dconv3, batch_norm8_6, relu, up, dconv4,
                             batch_norm8_7, relu, up, dconv5, batch_norm4_1,
                             relu, up, dconv7, tanh)

        if model_path is not None:
            self.load_weight(model_path)
            print("Load generator weight at {}".format(model_path))

            
    def forward(self,x):
        x = self.fc(x)
        x = x.unsqueeze(dim=2).unsqueeze(dim=3)
        out = self.model(x)
        return out

    def load_weight(self, path): 
        state_dict = torch.load(path)["state_dict"]
        fc_dict, model_dict = {}, {}
        for key, value in state_dict.items():
            if key.split(".")[1] != "decoder":
                continue
            new_key = key.replace("module.decoder.", "")
            if new_key == "fc.weight" or new_key == "fc.bias":
                fc_dict[new_key.replace("fc.", "")] = value
            else:
                model_dict[new_key.replace("model.", "")] = value
        self.fc.load_state_dict(fc_dict)
        self.model.load_state_dict(model_dict)

    
if __name__ == "__main__":
    model = Generator(model_path="/app/Talking_Head_Generation/src/external/PCL/checkpoints/pcl_test/4001.pth")
