import os
import random
import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, datapath, filename, training, transforms=None):
        self.datapath = datapath
        self.audios, self.labels = [], []
        self.filenames = []
        self.length = 250
        self.transforms = transforms
        self.training = training
        self.generator = random.Random(0)

        self.time_masking = T.TimeMasking(time_mask_param=96)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=24)

        self.load_path(filename)

        assert len(self.audios)==len(self.labels)
    
    def read_all_lines(self, filename):
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        return lines

    def load_path(self, list_filename):
        lines = self.read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        self.audios = [x[0] for x in splits]
        self.labels = [int(x[1]) for x in splits]

    def __len__(self):
        return len(self.audios)
    
    def __getitem__(self, idx):
        audio = self.audios[idx]
        label = self.labels[idx]

        audio = os.path.join(self.datapath, audio)
        values = np.load(audio).reshape(1, 128, self.length)
        values = torch.Tensor(values)
    
        if self.training:
            for i in range(len(values)):
                feat = values[i]
                if self.generator.random() > 0.6:
                    feat = self.time_masking(torch.unsqueeze(values[i], dim=0))
                if self.generator.random() > 0.6:
                    feat = self.freq_masking(feat)
                values[i] = feat

        values = values.repeat(3, 1, 1)

        if self.transforms:
            values = self.transforms(values)
        target = torch.LongTensor([label])
        return (values, target)

def fetch_dataloader(datapath, filename, batch_size, num_workers, training):
    dataset = AudioDataset(datapath, filename, training)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    return dataloader
