import torch
import os
from PIL import Image
from torchvision import transforms

class PACSDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, dataCats=[], transform=None):
        self.classMap = {
            "dog": 0,
            "elephant": 1,
            "giraffe": 2,
            "guitar": 3,
            "horse": 4,
            "house": 5,
            "person": 6,
        }
        self.root_dir = root_dir
        self.transform = transform

        self.data = []
        for cat in dataCats:
            self.data += self.getData(cat)

    def getData(self, category):    
        data = []

        path = os.path.join(self.root_dir, "images/" + category)
        for classDir in os.listdir(path):
            for imageFile in os.listdir(os.path.join(path, classDir)):
                data.append( [os.path.join(path, classDir, imageFile), self.classMap[classDir], category] )
        
        return data     

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(self.data[idx][0])
        if self.transform:
            image = self.transform(image)

        label = self.data[idx][1]

        return [image, label, self.data[idx][2]]
