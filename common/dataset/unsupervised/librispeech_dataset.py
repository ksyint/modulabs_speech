import sys
import torch
sys.path.append("../../../")
from common.utils import TextTransform
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import Wav2Vec2Processor

class LibriDataset(Dataset):
    def __init__(self, type:str, processor:str = "masoudmzb/wav2vec2-xlsr-multilingual-53-fa", padding="max_length", max_length:int = 392400):
        super(LibriDataset).__init__()
        self.librispeech = load_dataset("librispeech_asr", "clean", split=type)
        self.text_transform = TextTransform()
        self.processor = Wav2Vec2Processor.from_pretrained(processor)
        self.padding = padding
        self.max_length = max_length 

        # Train 100: 392400
        # Train 360: 475760
        # Validation: 522320
        # Test: 559280

    def __len__(self):
        return len(self.librispeech)
    
    def __getitem__(self, idx):
        data = self.librispeech[idx]
        features = self.processor(data["audio"]["array"], 
                        sampling_rate=self.processor.feature_extractor.sampling_rate, 
                        return_tensors="pt", padding="longest") #padding=self.padding, max_length=self.max_length)

        input_values = features.input_values 
        attention_mask = features.attention_mask 

        text = data["text"]
        token_id_str = " ".join(
            map(str, [_.item() for _ in self.text_transform.tokenize(text)])
        )
        token_id = torch.tensor([int(_) for _ in token_id_str.split()])

        return input_values.squeeze(0), attention_mask.squeeze(0), token_id

if __name__=="__main__":
    from torch.utils.data import DataLoader
    dataset = LibriDataset("test")
    input_values, attention_mask, token_id = dataset[0]
    print(input_values.shape, attention_mask.shape, token_id.shape)

    input_values, attention_mask, token_id = dataset[1]
    print(input_values.shape, attention_mask.shape, token_id.shape)
    dataloader = DataLoader(dataset, batch_size=1)
    max_len = 0
    len_list = []
    from tqdm import tqdm
    import numpy as np
    for batch in tqdm(dataloader):
        len_list.append(batch[2].shape[-1])
        if max_len < batch[2].shape[-1]:
            max_len = batch[2].shape[-1]
    print(max_len, np.mean(len_list))