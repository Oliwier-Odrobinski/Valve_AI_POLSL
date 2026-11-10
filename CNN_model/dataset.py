import os

import pandas as pd

from torch.utils.data import Dataset

from PIL import Image

import torch

import torchvision.transforms as transforms



class CustomImageDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):



        self.data_frame = pd.read_csv(csv_file, sep=';')

        self.img_dir = img_dir

        

        self.transform = transform if transform else transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize((0.5,0.5,0.5,0.5), (0.5,0.5,0.5,0.5))

        ])



        self.max_strumien = self.data_frame['strumien'].max()

        self.max_otwarcie = self.data_frame['otwarcie'].max()



    def __len__(self):

        return len(self.data_frame)

    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):

            idx = idx.tolist()

    

        row = self.data_frame.loc[idx]

    

        img_name = os.path.join(self.img_dir, row['nazwa'])

        image = Image.open(img_name)

    

        strumien = float(row['strumien'])

        otwarcie = float(row['otwarcie'])

    

        params = torch.tensor([

            strumien / self.max_strumien,

            otwarcie / self.max_otwarcie

        ], dtype=torch.float32)

    

        if self.transform:

            image = self.transform(image)

    

        return params, image

    

    

    def get_max_values(self):

        return self.max_strumien, self.max_otwarcie

